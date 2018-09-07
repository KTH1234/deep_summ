import torch
import torch.nn as nn
from torch.autograd import Variable

from onmt.Utils import aeq, sequence_mask, pad


class HierarchicalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.
    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.
    .. mermaid::
       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G
    All models compute the output as
    :math:`c = \sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].
    However they
    differ on how they compute the attention score.
    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`
    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`
    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]
    """
    def __init__(self, dim, coverage=False, attn_type="dot"):
        super(HierarchicalAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        
        # for intra-temporal attention save attn output history                
        self.attn_outputs = []
        # for intra-decoder attention save decoder output history        
        self.decoder_outputs = []
        
        assert (self.attn_type in ["dot", "general", "mlp"]), (
                "Please select a valid attention type.")

        if self.attn_type == "general":
            self.sent_linear_in = nn.Linear(dim, dim, bias=False)
            self.context_linear_in = nn.Linear(dim, dim, bias=False)
            # weight matrix for intra-decoder attention
#             self.linear_in_intra_decoder = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        
        # concat 3 vector : decoder output, temporal attention, decoder attention
        self.linear_out = nn.Linear(dim*2, dim, bias=out_bias)

        self.sm = nn.Softmax()
        self.logsm = nn.LogSoftmax()
        self.tanh = nn.Tanh()

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

        # for intra-temporal attention, init attn history per every batches
    def init_attn_outputs(self):
        self.sent_attn_outputs = None
        self.sent_attn_outputs = []
        
        self.context_attn_outputs = None
        self.context_attn_outputs = []        
#         print("gb attn line:98, len attn_otputs", len(self.attn_outputs))
            
    # for intra-decoder attention, init decoder output history
    def init_decoder_outputs(self):
        self.sent_decoder_outputs = None
        self.sent_decoder_outputs = []
        
        self.context_decoder_outputs = None
        self.context_decoder_outputs = []        
#         print("gb attn line:103, len decoder_outputs", len(self.decoder_outputs))

    def score(self, h_t, h_s, typ="context"):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`
        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`
        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
                
                # use seperate weight matrix for intra decoder and temporal attention
                if typ == "context":
                    h_t_ = self.context_linear_in(h_t_)
                elif type == "sent":
                    h_t_ = self.sent_linear_in(h_t_)
#                 else:
#                     h_t_ = self.linear_in_intra_decoder(h_t_)
                   
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)
        
    def get_alignment(self, input, memory_bank, memory_lengths, attn_outputs, coverage=None, emb_weight=None, idf_weights=None, no_intra_attn=False, no_dec_attn=False, type="context"):
        """
        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)
          
          # thkim
          emb_weight : maybe intra attention related ...
          idf_weights : idf values, multiply it to attn weight
        Returns:
          (`FloatTensor`, `FloatTensor`):
          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """
        assert memory_lengths is not None

        # one step input
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False

        batch, sourceL, dim = memory_bank.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, sourceL_ = coverage.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank += self.linear_cover(cover).view_as(memory_bank)
            memory_bank = self.tanh(memory_bank)

        # compute attention scores, as in Luong et al.
        align = self.score(input, memory_bank)
        
        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths.data)
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.data.masked_fill_(1 - mask, -float('inf'))
                                                                                                                                                                                                                                                                                                                 
        ## Intra-temporal attention
        ## assum train is going on the gpu    
        
        align = torch.exp(align) # batch * 1(target_length) * input_length
#         print("globalattn line 203: align")

#         align_vectors = self.logsm(align.view(batch*targetL, sourceL))
        align_vectors = self.sm(align.view(batch*targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)
#         if len(attn_outputs) < 1: # t=1
# #             print("global attn line:208, attn_outputs")
# #             print(len(self.attn_outputs))
#             align_vectors = self.logsm(align.view(batch*targetL, sourceL))
# #             align_vectors = self.sm(align.view(batch*targetL, sourceL))
    
    
#             align_vectors = align_vectors.view(batch, targetL, sourceL)
#         else: # t > 1
# #             print("hier attn line:209, attn_outputs", attn_outputs)
# #             print(len(self.attn_outputs))
#             temporal_attns = torch.cat(attn_outputs, 1) # batch * len(t-1) * input_length
#             normalizing_factor = torch.sum(temporal_attns,1).unsqueeze(1)
# #             print("global attn line:214, normalizing factor")

#             # wrong implementation 
#             # normalizing_factor = torch.autograd.Variable(torch.cat([torch.ones(align.size()[0], 1, 1).cuda(), torch.cumsum(torch.exp(align), 2).data[:,:,:-1]],2))
# #             align = torch.exp(align) / normalizing_factor
# #             align_vectors = align / torch.sum(align, 2).unsqueeze(2)            
            
#             align_vectors = align / normalizing_factor            
#             align_vectors = self.logsm(align.view(batch*targetL, sourceL))
# #             align_vectors = self.sm(align.view(batch*targetL, sourceL))
#             align_vectors = align_vectors.view(batch, targetL, sourceL)

        # Softmax to normalize attention weights
        ## 기존 attention
#         align_vectors = self.sm(align.view(batch*targetL, sourceL))
#         align_vectors = align_vectors.view(batch, targetL, sourceL)


#         print("global attn line:270 idf_weights", torch.autograd.Variable(idf_weights.t().unsqueeze(1), requires_grad=False))
#         print("global attn line:270", align_vectors)
        if idf_weights is not None:
            align_vectors = align_vectors * torch.autograd.Variable(idf_weights.t().unsqueeze(1), requires_grad=False)
#         input()

        # each context vector c_t is the weighted average
        # over all the source hidden states

#         attn_outputs.append(align)
#         print("gb attn line:237 len attn_outputs", len(self.attn_outputs))
        
        # sent attn no need to dec attn at this time
        if no_dec_attn:
            return align_vectors, None
        
        return align_vectors
  

    def forward(self, input, sentence_memory_bank, context_memory_bank, sentence_memory_lengths, context_memory_lengths, context_mask, coverage=None, emb_weight=None, idf_weights=None):
        """
        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)
          
          # thkim
          emb_weight : maybe intra attention related ...
          idf_weights : idf values, multiply it to attn weight
        Returns:
          (`FloatTensor`, `FloatTensor`):
          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """
        
        # one step input
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False
        
        batch_, targetL, dim = input.size()
        batch = batch_
            
        #############################################################################            
        # ======== intra-decoder attention
#         if len(self.context_decoder_outputs,) < 1:
# # TO DO : change initial value to zero vector
# # ? what is size of zero vector? 밑에 decoder attn도 조금 이상해 보임
#             # set zero vector to first case
#             c_dec = input * 0 
            
# #             self.context_decoder_outputs.append(input)      
# #             print("glbal-attn", "dd")
#         else:
#             decoder_history = torch.cat(self.context_decoder_outputs, 1) # batch * tgt_len(?) * dim
#             decoder_align = self.score(input, decoder_history, "dec_attn")
# #             print("global attn line:223 decoder align")
# #             print(decoder_align)
# #             input()

# #             print("global-attn line:225", decoder_history)
# #             if len(self.decoder_outputs) == 5:
# #                 input()
            
#             history_len = len(self.context_decoder_outputs)
#             decoder_align_vectors = self.sm(decoder_align.view(batch*targetL, history_len))
#             decoder_align_vectors = decoder_align_vectors.view(batch, targetL, history_len)
#             c_dec = torch.bmm(decoder_align_vectors, decoder_history)
# #             print("global-attn line:232", decoder_align_vectors) 

#         self.context_decoder_outputs.append(input) 
        #############################################################################
#         print("hiera attn line:332 sentence_memory_bank",sentence_memory_bank.size()) # batch * max_src__len * hidden
#         print("hiera attn line:333 input",input.size()) # batch * 1 * context len
#         print("hiera attn line:333 context_memory_lengths", context_memory_lengths) # batch
#         print("hiera attn line:333 context_mask", context_mask) # batch
        
        # context_align_vector : batch * tgt len * context len
        context_align_vector = self.get_alignment(input, context_memory_bank, context_memory_lengths, self.context_attn_outputs, coverage=None, emb_weight=None, idf_weights=None, type="context")
  
        # sent_context_input : sum(context_lengths) * 1 * hidden
        sent_context_input = torch.cat([ input[batch_idx].expand(context_len, input.size(1), input.size(2))  for batch_idx, context_len in enumerate(context_memory_lengths.data) ])
#         print("hiera attn line:339 context_memory_lengths", sent_context_input) # batch * 1 * context len
        
        # sent_align_vectors : batch * tgt len * max_src len
        sent_align_vectors = self.get_alignment(sent_context_input, sentence_memory_bank, sentence_memory_lengths, self.sent_attn_outputs, coverage=None, emb_weight=None, idf_weights=None, type="sent")
        
        # context_attn_mask : batch
        context_attn_mask = torch.ones(context_align_vector.size(0), context_align_vector.size(2)).cuda() # batch * max_context_len
        max_context_len = torch.max(context_memory_lengths.data)
        
        for batch_idx, context_len in enumerate(context_memory_lengths.data):
            diff = max_context_len - context_len
            if diff == 0:
                continue
            else:
                context_attn_mask[batch_idx][-diff:] = 0
        #print("hiera attn line:356 context_attn_mask", context_attn_mask) # batch
        
        # select real context position in mask
        # flatten_context_align_vector : sum(context_length)
        
#        if input != input:
#            print("hiera attn line:324 nan input", input) # batch * 1 * sent lenz
        
        flatten_context_align_vector = context_align_vector.squeeze(1)[context_attn_mask == 1]
                
        # calculate last attn
#         sent_align_vectors = sent_align_vectors + flatten_context_align_vector.view(-1,1,1)
        sent_align_vectors = sent_align_vectors * flatten_context_align_vector.view(-1,1,1)
#        print("hiera attn line:324 sent_align_vector", sent_align_vectors) # batch * 1 * sent lenz
#        input()        
        # pad
#         if sentence_memory_lengths is not None:
#             mask = sequence_mask(sentence_memory_lengths.data)
#             mask = mask.unsqueeze(1)  # Make it broadcastable.

        if torch.sum(sent_align_vectors != sent_align_vectors).data[0] != 0:
            print("hiera attn line:387 sent_align_vectors",sent_align_vectors) #
        
        # prev 18-08-17
#         sent_align_vectors = self.sm(sent_align_vectors.squeeze(1)).unsqueeze(1)
        
        if torch.sum(sent_align_vectors != sent_align_vectors).data[0] != 0:
            print("hiera attn line:385 context_memory_bank",context_memory_bank) #
            print("hiera attn line:386 sentence_memory_bank",sentence_memory_bank) #
            print("hiera attn line:387 input",input) #
            print("hiera attn line:387 sent_align_vectors",sent_align_vectors) #
            print("hiera attn line:387 context_align_vector",context_align_vector) #
            
            if torch.sum(context_align_vector != context_align_vector).data[0] != 0:
                print("hiera attn line:387 context_align_vector contains nan") #
            input()
        
#        print("hiera attn line:322 context_memory_bank",context_memory_bank) # batch * 1 * context len
#        print("hiera attn line:322 flatten_context_align_vector",flatten_context_align_vector) # batch * 1 * context len
        #print("hiera attn line:322 context_align_vector",context_align_vector) # batch * 1 * context len
#        print("hiera attn line:322 context_align_vector",context_align_vector.size()) # batch * 1 * context len
#         print("hiera attn line:323 context_decoder", context_decoder_align_vector) 
#        print("hiera attn line:324 sent_align_vector", sent_align_vectors) # batch * 1 * sent lenz


        # context_attn_mask : batch
        sent_attn_mask = torch.ones(sent_align_vectors.size(0), sent_align_vectors.size(2)).cuda() # batch * max_context_len
        sent_memory_mask = torch.ones(sentence_memory_bank.size()).cuda() # batch * max_context_len
        max_sentence_len = torch.max(sentence_memory_lengths.data)
#         print("hiera attn line:378 max_sentence_len", max_sentence_len)
        
        for batch_idx, sent_len in enumerate(sentence_memory_lengths.data):
            diff = max_sentence_len - sent_len
            if diff == 0:
                continue
            else:
                sent_attn_mask[batch_idx][-diff:] = 0
                sent_memory_mask[batch_idx][-diff:] = 0
#         print("hiera attn line:356 sent_attn_mask", sent_attn_mask) # batch
#         print("hiera attn line:389 sent_memory_mask", sent_memory_mask) # batch
        
        # select real context position in mask
        # flatten_context_align_vector : sum(context_length)
        flatten_sent_align_vector = sent_align_vectors.squeeze(1)[sent_attn_mask == 1]
        sentence_memory_bank = sentence_memory_bank[sent_memory_mask == 1].view(flatten_sent_align_vector.size(0),dim)
        flatten_sent_memory_vector = sentence_memory_bank
#         print("hiera attn line:391 flatten_sent_align_vector", flatten_sent_align_vector.size()) # batch
#         print("hiera attn line:391 flatten_sent_memory_vector", flatten_sent_memory_vector.size()) # batch
        
        global_sentence_memory_length = torch.sum((context_mask >= 0).long(), 0) # batch
#         print("hiera attn line:394 global_sentence_memory_length", global_sentence_memory_length) # batch
        sub_global_sent_memory_length = global_sentence_memory_length.gather(0, Variable(torch.LongTensor(range(batch-1))).cuda())
#         print("hiera attn line:394 sub_global_sent_memory_length", sub_global_sent_memory_length) # batch
        max_sentence_length = torch.max(global_sentence_memory_length)
        
        sent_start_index = torch.cumsum(torch.cat([torch.zeros(1).long().cuda(), sub_global_sent_memory_length.data]), 0)
#         print("hiera attn line:394 sent_start_index", sent_start_index) # batch
         
        flatten_sent_align_vector = torch.stack([ pad(flatten_sent_align_vector.narrow(0, s, l), max_sentence_length.data[0], pad_index=-float('inf')) for s, l in zip(sent_start_index, global_sentence_memory_length.data) ]).unsqueeze(1)
        flatten_sent_memory_vector = torch.stack([ pad(flatten_sent_memory_vector.narrow(0, s, l), max_sentence_length.data[0]) for s, l in zip(sent_start_index, global_sentence_memory_length.data) ])
#         print("hiera attn line:391 flatten_sent_align_vector", flatten_sent_align_vector.size()) # batch * 1 * max_src_len
#         print("hiera attn line:391 flatten_sent_memory_vector", flatten_sent_memory_vector.size()) # batch * * max_src_len * dim

        flatten_sent_align_vector = self.sm(flatten_sent_align_vector.squeeze(1)).unsqueeze(1)        
    
        sent_align_vector = flatten_sent_align_vector
        batch, sourceL, dim = flatten_sent_memory_vector.size()
#         batch, sourceL, dim = sentence_memory_bank.size()
#         batch_, targetL, dim_ = input.size()
#         aeq(batch, batch_)
#         aeq(dim, dim_)
#         aeq(self.dim, dim)
#         if coverage is not None:
#             batch_, sourceL_ = coverage.size()
#             aeq(batch, batch_)
#             aeq(sourceL, sourceL_)

#         if coverage is not None:
#             cover = coverage.view(-1).unsqueeze(1)
#             memory_bank += self.linear_cover(cover).view_as(memory_bank)
#             memory_bank = self.tanh(memory_bank)       

#         print("hiera attn line:390 sent_align_vectors", sent_align_vectors.size()) # batch * 1 * sent lenz        
#         print("hiera attn line:390 sentence_memory_bank", sentence_memory_bank.size()) # batch * 1 * sent lenz        
#         c = torch.bmm(sent_align_vectors, sentence_memory_bank) # for intra-temporal attention
        c = torch.bmm(flatten_sent_align_vector, flatten_sent_memory_vector) # for intra-temporal attention
#         print("hiera attn line:390 c", c.size()) # batch * 1 * sent lenz        
#         print("hiera attn line:390 input", input.size()) # batch * 1 * sent lenz        
#         print("hiera attn line:390 c_dec", c_dec.size()) # batch * 1 * sent lenz        
#         input("hier attn:line:348")        
#         if context_decoder_align_vector is None:
#             c_dec = input * 0 
#         else:
#             decoder_history = torch.cat(self.context_decoder_outputs, 1) # batch * tgt_len(?) * dim
#             decoder_align = self.score(input, decoder_history, "dec_attn")
# #             print("global attn line:223 decoder align")
# #             print(decoder_align)
# #             input()

# #             print("global-attn line:225", decoder_history)
# #             if len(self.decoder_outputs) == 5:
# #                 input()
            
#             history_len = len(self.context_decoder_outputs)
#             decoder_align_vectors = self.sm(decoder_align.view(batch*targetL, history_len))
#             decoder_align_vectors = decoder_align_vectors.view(batch, targetL, history_len)
# #             print("global-attn line:232", decoder_align_vectors) 
#             c_dec = torch.bmm(decoder_align_vectors, decoder_history)  
    
#         print("h attn line:371 c", c.size())
#         print("h attn line:372 input", input.size())
#         print("h attn line:372 c_dec", c_dec.size())
#         input()

        # concatenate
#         concat_c = torch.cat([c, input, c_dec], 2).view(batch*targetL, dim*3)
        concat_c = torch.cat([c, input], 2).view(batch*targetL, dim*2)
#         concat_c = torch.cat([c, input], 2).view(batch*targetL, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        
        
        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            sent_align_vector = sent_align_vector.squeeze(1)
            context_align_vector = context_align_vector.squeeze(1)
            
            
            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, sourceL_ = sent_align_vector.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
            batch_, contextL_ = sent_align_vector.size()
            aeq(batch, batch_)
            aeq(max_sentence_length.data[0], contextL_)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            sent_align_vector = sent_align_vector.transpose(0, 1).contiguous()

            # Check output sizes
            targetL_, batch_, dim_ = attn_h.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            targetL_, batch_, sourceL_ = sent_align_vector.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
#         print("hier attn line:618 attn_h", attn_h.requires_grad)
#         print("hier attn line:618 sent_align_vector", sent_align_vector.requires_grad)
#         print("hier attn line:618 context_align_vector", context_align_vector.requires_grad)
#         input("hier attn line:610")
        return attn_h, sent_align_vector, context_align_vector
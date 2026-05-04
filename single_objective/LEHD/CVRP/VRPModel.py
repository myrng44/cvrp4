import torch
import torch.nn as nn
import torch.nn.functional as F


class VRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']
        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)

        self.encoded_nodes = None

    def forward(self, state, selected_node_list, solution, current_step,raw_data_capacity=None,):
        # solution's shape : [B, V]
        self.capacity = raw_data_capacity.ravel()[0].item()
        batch_size = state.problems.shape[0]
        problem_size = state.problems.shape[1]
        split_line = problem_size - 1

        def probs_to_selected_nodes(probs_,split_line_,batch_size_):
            selected_node_student_ = probs_.argmax(dim=1)  # shape: B
            is_via_depot_student_ = selected_node_student_ >= split_line_ # Nodes with an index greater than customer_num are via depot
            not_via_depot_student_ = selected_node_student_ < split_line_

            selected_flag_student_ = torch.zeros(batch_size_,dtype=torch.int)
            selected_flag_student_[is_via_depot_student_] = 1
            selected_node_student_[is_via_depot_student_] = selected_node_student_[is_via_depot_student_]-split_line_ +1
            selected_flag_student_[not_via_depot_student_] = 0
            selected_node_student_[not_via_depot_student_] = selected_node_student_[not_via_depot_student_]+ 1
            return selected_node_student_, selected_flag_student_ # node 的 index 从 1 开始

        if self.mode == 'train':
            remaining_capacity = state.problems[:, 1, 3]

            probs = self.decoder(self.encoder(state.problems,self.capacity),
                                 selected_node_list, self.capacity,remaining_capacity, state.problems)

            selected_node_student, selected_flag_student = probs_to_selected_nodes(probs, split_line, batch_size)

            selected_node_teacher = solution[:, current_step,0]

            selected_flag_teacher = solution[:, current_step, 1]

            is_via_depot = selected_flag_teacher==1
            selected_node_teacher_copy = (selected_node_teacher-1).long()
            selected_node_teacher_copy[is_via_depot]+=split_line
            # print('selected_node_teacher after',selected_node_teacher)
            prob_select_node = probs[torch.arange(batch_size)[:, None], selected_node_teacher_copy[:, None]].reshape(batch_size, 1)  # shape: [B, 1]

            loss_node = -prob_select_node.type(torch.float64).log().mean()

        if self.mode == 'test':

            remaining_capacity = state.problems[:, 1, 3]
            # print(state.problems.shape)
            if current_step <= 1:
                self.encoded_nodes = self.encoder(state.problems,self.capacity)


            probs = self.decoder(self.encoded_nodes, selected_node_list,self.capacity, remaining_capacity, state.problems)

            selected_node_student = probs.argmax(dim=1)  # shape: B
            is_via_depot_student = selected_node_student >= split_line  # 节点index大于 customer_num的是通过depot的
            not_via_depot_student = selected_node_student < split_line
            # print(selected_node_student)
            selected_flag_student = torch.zeros(batch_size, dtype=torch.int)
            selected_flag_student[is_via_depot_student] = 1
            selected_node_student[is_via_depot_student] = selected_node_student[is_via_depot_student] - split_line + 1
            selected_flag_student[not_via_depot_student] = 0
            selected_node_student[not_via_depot_student] = selected_node_student[not_via_depot_student] + 1

            selected_node_teacher = selected_node_student
            selected_flag_teacher = selected_flag_student

            loss_node = torch.tensor(0)

        return loss_node,selected_node_teacher,  selected_node_student,selected_flag_teacher,selected_flag_student




class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num =  3
        self.embedding = nn.Linear(3, embedding_dim, bias=True)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data_,capacity):

        data = data_.clone().detach()
        data= data[:,:,:3]

        # === AAFM: compute pairwise distance from (x,y) coordinates ===
        coords = data[:, :, :2]  # (B, n, 2)
        dist = torch.cdist(coords, coords)  # (B, n, n)

        data[:,:,2] = data[:,:,2]/capacity

        embedded_input = self.embedding(data)

        out = embedded_input  # [B*(V-1), problem_size - current_step +2, embedding_dim]

        layer_count = 0
        for layer in self.layers:
            out = layer(out, dist)  # pass dist for AAFM
            layer_count += 1
        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        # === AAFM: project to embedding_dim directly (no head splitting) ===
        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.alpha = nn.Parameter(torch.Tensor([1.]), requires_grad=True)  # learnable scale

        self.feedForward = Feed_Forward_Module(**model_params)


    def forward(self, input1, dist):
        # input1: (B, n, embedding_dim), dist: (B, n, n)

        q = self.Wq(input1)
        k = self.Wk(input1)
        v = self.Wv(input1)

        # === AAFM with distance-based adaptation_bias ===
        adaptation_bias = -self.alpha * dist  # (B, n, n)
        aafm_out = adaptation_attention_free_module(q, k, v, adaptation_bias)

        out1 = input1 + aafm_out
        out2 = self.feedForward(out1)

        out3 = out1 + out2
        return out3
        # shape: (batch, problem, EMBEDDING_DIM)

########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        decoder_layer_num = self.model_params['decoder_layer_num']

        self.embedding_first_node = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node = nn.Linear(embedding_dim+1, embedding_dim, bias=True)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(decoder_layer_num)])
        self.Linear_final = nn.Linear(embedding_dim, 2, bias=True)

    def _get_new_data(self, data, selected_node_list, prob_size, B_V):

        list = selected_node_list

        new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1)

        new_list_len = prob_size - list.shape[1]  # shape: [B, V-current_step]

        index_2 = list.type(torch.long)

        index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2.shape[1])

        new_list[index_1, index_2] = -2

        unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)

        new_data = data

        emb_dim = data.shape[-1]

        new_data_len = new_list_len

        index_2_ = unselect_list.repeat_interleave(repeats=emb_dim, dim=1)

        index_1_ = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2_.shape[1])

        index_3_ = torch.arange(emb_dim)[None, :].repeat(repeats=(B_V, new_data_len))

        new_data_ = new_data[index_1_, index_2_, index_3_].view(B_V, new_data_len, emb_dim)

        return new_data_

    def _get_encoding(self,encoded_nodes, node_index_to_pick):

        batch_size = node_index_to_pick.size(0)
        pomo_size = node_index_to_pick.size(1)
        embedding_dim = encoded_nodes.size(2)

        gathering_index = node_index_to_pick[:, :, None].long().expand(batch_size, pomo_size, embedding_dim)

        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

        return picked_nodes


    def forward(self, data,selected_node_list,capacity,remaining_capacity, raw_data):

        data_ = data[:,1:,:].clone().detach()
        selected_node_list_ = selected_node_list.clone().detach() - 1

        batch_size_V = data_.shape[0]  # B

        problem_size = data_.shape[1]

        new_data = data_.clone().detach()

        left_encoded_node = self._get_new_data(new_data, selected_node_list_, problem_size, batch_size_V)

        embedded_first_node = data[:,[0],:]

        if selected_node_list_.shape[1]==0:
            embedded_last_node = data[:,[0],:]
        else:
            embedded_last_node = self._get_encoding(new_data, selected_node_list_[:, [-1]])

        remaining_capacity = remaining_capacity.reshape(batch_size_V,1,1)/capacity
        first_node_cat = torch.cat((embedded_first_node,remaining_capacity), dim=2)
        last_node_cat = torch.cat((embedded_last_node,remaining_capacity), dim=2)
        # ------------------------------------------------
        # ------------------------------------------------

        embedded_first_node_ = self.embedding_first_node(first_node_cat)

        embedded_last_node_ = self.embedding_last_node(last_node_cat)


        embeded_all = torch.cat((embedded_first_node_,left_encoded_node,embedded_last_node_), dim=1)
        out = embeded_all  # [B*(V-1), problem_size - current_step +2, embedding_dim]

        # === AAFM: compute distance matrix for assembled sequence ===
        # Coordinates: [depot, remaining_customers, last_node]
        all_customer_coords = raw_data[:, 1:, :2]  # (B, problem_size, 2)
        depot_coords = raw_data[:, [0], :2]  # (B, 1, 2)
        left_coords = self._get_new_data(all_customer_coords, selected_node_list_, problem_size, batch_size_V)
        if selected_node_list_.shape[1] == 0:
            last_coords = raw_data[:, [0], :2]
        else:
            last_coords = self._get_encoding(all_customer_coords, selected_node_list_[:, [-1]])
        all_coords = torch.cat([depot_coords, left_coords, last_coords], dim=1)  # (B, n, 2)
        dist = torch.cdist(all_coords, all_coords)  # (B, n, n)

        layer_count = 0

        for layer in self.layers:

            out = layer(out, dist)  # pass dist for AAFM
            layer_count += 1


        out = self.Linear_final(out)  # shape: [B*(V-1), reminding_nodes_number + 2, embedding_dim ]

        out[:, [0, -1], :] = out[:, [0, -1], :] + float('-inf')  # first node、last node

        out = torch.cat((out[:, :, 0], out[:, :, 1]), dim=1)  # shape:(B, 2 * ( V - current_step ))

        props = F.softmax(out, dim=-1)
        customer_num = left_encoded_node.shape[1]

        props = torch.cat((props[:, 1:customer_num + 1], props[:, customer_num + 1 + 1 + 1:-1]),
                          dim=1)

        index_small = torch.le(props, 1e-5)
        props_clone = props.clone()
        props_clone[index_small] = props_clone[index_small] + torch.tensor(1e-7, dtype=props_clone[index_small].dtype)
        props = props_clone

        new_props = torch.zeros(batch_size_V, 2 * (problem_size))

        # The function of the following part is to fill the probability of props into the new_props,
        index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:,None].repeat(1,selected_node_list_.shape[1]*2)
        index_2_ =torch.cat( ((selected_node_list_).type(torch.long), (problem_size)+ (selected_node_list_).type(torch.long) ),dim=-1) # shape: [B*V, n]
        new_props[index_1_, index_2_,] = -2
        index = torch.gt(new_props, -1).view(batch_size_V, -1)
        new_props[index] = props.ravel()

        return new_props


class DecoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        # === AAFM: project to embedding_dim directly (no head splitting) ===
        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.alpha = nn.Parameter(torch.Tensor([1.]), requires_grad=True)  # learnable scale
        self.feedForward = Feed_Forward_Module(**model_params)

    def forward(self, input1, dist):
        # input1: (B, n, embedding_dim), dist: (B, n, n)

        q = self.Wq(input1)
        k = self.Wk(input1)
        v = self.Wv(input1)

        # === AAFM with distance-based adaptation_bias ===
        adaptation_bias = -self.alpha * dist  # (B, n, n)
        aafm_out = adaptation_attention_free_module(q, k, v, adaptation_bias)

        out1 = input1 + aafm_out
        out2 = self.feedForward(out1)
        out3 = out1 + out2
        return out3



def adaptation_attention_free_module(q, k, v, adaptation_bias=None, ninf_mask=None):
    """
    AAFM (Adaptive Attention Free Module) from ICAM.
    Replaces traditional multi-head attention.

    Args:
        q: query, shape: (batch, n, embedding_dim)
        k: key,   shape: (batch, m, embedding_dim)
        v: value, shape: (batch, m, embedding_dim)
        adaptation_bias: optional, shape: (batch, n, m). Default None (no bias).
        ninf_mask: optional, shape: (batch, n, m)

    Returns:
        out: shape: (batch, n, embedding_dim)
    """
    sigmoid_q = torch.sigmoid(q)
    # shape: (batch, n, embedding_dim)

    exp_k = torch.exp(k)
    # shape: (batch, m, embedding_dim)

    if adaptation_bias is not None or ninf_mask is not None:
        bias_term = torch.zeros(q.size(0), q.size(1), k.size(1), device=q.device, dtype=q.dtype)
        if adaptation_bias is not None:
            bias_term = bias_term + adaptation_bias
        if ninf_mask is not None:
            bias_term = bias_term + ninf_mask
        exp_bias = torch.exp(bias_term)
        # shape: (batch, n, m)
        numerator = exp_bias @ torch.mul(exp_k, v)
        denominator = exp_bias @ exp_k
    else:
        # No bias: simplifies to AFT-simple
        numerator = torch.mul(exp_k, v).sum(dim=1, keepdim=True).expand_as(q)
        denominator = exp_k.sum(dim=1, keepdim=True).expand_as(q)

    weighted = numerator / denominator
    if torch.isinf(numerator).any() or torch.isinf(denominator).any():
        weighted = torch.nan_to_num_(numerator) / torch.nan_to_num_(denominator)
    if torch.isnan(weighted).any():
        torch.nan_to_num_(weighted)
    # shape: (batch, n, embedding_dim)

    out = torch.mul(sigmoid_q, weighted)
    # shape: (batch, n, embedding_dim)

    return out


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):


        return self.W2(F.relu(self.W1(input1)))

import math
from re import A
from turtle import forward
from typing import Optional
import torch
from torch import  nn
from torch.utils.data import Dataset
from configuration_antibody import *
import pdb
from packaging import version
import math
from activations import ACT2FN
from modeling_outputs import *
import warnings
import numpy as np

def adjust_tensors_for_parallel(hidden_states, *tensors):
    """
    Replicates a given list of tensors based on the shape of the reference tensor (first argument).
    """
    outputs = []
    for tensor in tensors:
        if tensor is not None and hidden_states.shape[0] != tensor.shape[0]:
            repeats = [1] * len(tensor.shape)
            repeats[0] = hidden_states.shape[0] // tensor.shape[0]
            new_tensor = tensor.repeat(*repeats)
            outputs.append(new_tensor)
        else:
            outputs.append(tensor)
    return tuple(outputs)

def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    pad_attn_mask =  pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
    mask = torch.zeros(pad_attn_mask.shape)
    mask[pad_attn_mask] = torch.finfo(torch.float).min
    return mask.unsqueeze(1)

def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = (subsequence_mask == 1)
    subsequence_mask = torch.from_numpy(subsequence_mask)
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    pad_attn_mask =  pad_attn_mask.expand(attn_shape)  # [batch_size, len_q, len_k]
    mask = torch.zeros(subsequence_mask.shape)
    mask[subsequence_mask|pad_attn_mask] = torch.finfo(torch.float).min
    #subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return mask.unsqueeze(1) # [batch_size, tgt_len, tgt_len]

class AntiEmbeddings(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        """Construct the embeddings from word, position and token_type embeddings.
        residue、 position、 token_type"""
        self.config = config
        self.residue_embedding = nn.Embedding(config.token_size,config.hidden_size,padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,config.hidden_size)
        if config.type_embedding == True:
            self.token_type_embeddings = nn.Embedding(config.type_residue_size,config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size,eps = config.layer_norm_eps)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        ## register_buffer data will not be update after backward
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )
    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        ):
        
        '''
        parameters:
            input_ids : residue - id (VH or VL)
            token_type_ids : CDR(n) or framework
            position_ids : the index of sequence (VH or VL)
            inputs_embeds : using other ways to embedding        **[optional]**
            past_key_values_length : defatult key in position_id **[optional]**
        return embedding
        '''
        
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        ## residue sequence embedding
        if inputs_embeds is None:
            if self.config.one_hot:
                embeddings = input_ids @ self.residue_embedding.weight
            else:
                embeddings = self.residue_embedding(input_ids)

        ## token_type embedding
        if self.config.type_embedding == True:
            
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
        

class AntiSelfAttention(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size,self.all_head_size,bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size,self.all_head_size,bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size,self.all_head_size,bias=config.use_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self,x):
        """
        return [batch, heads, sequence, Hidden_size]
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape) # x [batch, sequence, heads, Hiddensize]

        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)
            

        # Take the dot product between "query" and "key" to get the raw attention scores.
      
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BigBirdModel forward() function)
            
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        ### attention scores
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class AntiSelfOutput(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states:torch.tensor, input_tensor:torch.Tensor):
        """
        hindden states -> hidden states
                            + input     (resnet)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class AntiAttention(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        """
        choose attention
        [option]: original_full ## transofrmer
        """
        if config.attention_type == 'original_full':
            self.self = AntiSelfAttention(config=config)
        else:
            raise ValueError(f"attention_type can not be figure out")
        self.output = AntiSelfOutput(config)
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                output_attentions: Optional[bool] = False,
                )-> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
    
class AntiAttentionSegment(nn.Module):
    def __init__(self,config):
        super().__init__()
        """
        choose attention
        [option]: original_full ## transofrmer
        """
        self.Blord = nn.Linear(config.hidden_size,config.hidden_size) # map Big lord 
        if config.attention_type == 'original_full':
            self.self = []
            for _ in range(config.segments):
                self.self.append(AntiSelfAttention(config=config))
            self.self = nn.ModuleList(self.self)
            self.lords = AntiSelfAttention(config=config)             # lords attention
        else:
            raise ValueError(f"attention_type can not be figure out")
        self.output = AntiSelfOutput(config)
        self.config = config
    def forward(self,
                hidden_states: torch.Tensor,
                x:torch.Tensor = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                output_attentions: Optional[bool] = False,
                )-> Tuple[torch.Tensor]:
        Blord = self.Blord(hidden_states[:,0])
        self_outputs = [Blord.unsqueeze(1)]
        ## segment attention
        #temp = []
        
        for i in range(self.config.segments):
            #[:, self.config.segment_size*i+1+i:self.config.segment_size*(i+1)+1+i+1]
            # get block atteion mask
            
            # temp = x[:,self.config.segment_size*i+1+i:self.config.segment_size*(i+1)+1+i+1]
            # t_attention_mask = get_attn_pad_mask(temp,temp).cuda()
            
            ## get block atteinon
            self_output = self.self[i](
                hidden_states[:, self.config.segment_size*i+1+i:self.config.segment_size*(i+1)+1+i+1],  # NOTE mask not add
                attention_mask = attention_mask if attention_mask == None else attention_mask[:,:,self.config.segment_size*i+1+i:self.config.segment_size*(i+1)+1+i+1,self.config.segment_size*i+1+i:self.config.segment_size*(i+1)+1+i+1]
            )
            self_outputs.append(self_output[0])
        # lords attention
        
        lords_id = [0]
        lords_id.extend([i*self.config.segment_size+1 + i for i in range(self.config.segments)])
        lords = [lord[:,0,:].unsqueeze(1) for lord in self_outputs]
        lords = torch.cat(lords,dim=1)
        
        
        # get lords mask
        # temp = x[:, lords_id]
        # t_attention_mask = get_attn_pad_mask(temp,temp).cuda()
        # get lords attention
        
        lords = self.lords(lords,                                                                  
        attention_mask if attention_mask == None else attention_mask[:,:,lords_id][:,:,:,lords_id]
                            )[0]
        
        temp_outputs = [lords[:,0,:].unsqueeze(1)]
        for i in range(1,self.config.segments+1):
            temp_outputs.append(lords[:,i].unsqueeze(1))
            temp_outputs.append(self_outputs[i][:,1:,:])
        
        self_outputs = torch.cat(temp_outputs,dim=1)
        attention_output = self.output(self_outputs, hidden_states)
        outputs = (attention_output,) + (self_outputs,)  # add attentions if we output them
        
        return outputs

class AntiIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        intermeiate_size = 4 * hidden_size
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class AntiDownSample(nn.Module):
    def __init__(self,config,origin,down) -> None:
        super().__init__()
        self.origin = origin
        self.down = down
        self.dense1 = nn.Linear(self.origin,self.down*4)
        self.dense2 = nn.Linear(self.down*4,self.down)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn1 = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn1 = config.hidden_act

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn2 = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn2 = config.hidden_act

    def forward(self,hidden_states: torch.Tensor) -> torch.Tensor:
        
        hidden_states = hidden_states.permute(0,2,1)
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn1(hidden_states) 

        hidden_states = self.dense2(hidden_states)
        hidden_states = self.intermediate_act_fn2(hidden_states) 
        hidden_states = hidden_states.permute(0,2,1) 
        return hidden_states

class AntiOutput(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size,config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    

class AntiLayer(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config
        self.attention_typ = config.attention_type
        self.attention = AntiAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise TypeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = AntiAttention(config)
        self.intermediate = AntiIntermediate(config)
        self.output = AntiOutput(config)
    
    def forward(self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
                ):
        """
        paramterts 

        hidden_states : residue hidden states [batch , length , hidden size ] 

        NOTE : optional paramete
        attention_mask :   , mask padding token
        head_mask : , Mask heads if we want to 
        encoder_hidden_states :  if Layer is decoder, the parameters is expeted. the expression of framework.
        encdoer_attention_mask : mask framework's padding token or other special token
        pask_key_value : hidden_states as query   
        output_attentions : whether get attention score matrix
        """
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )
          
            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_out = self.feed_forward(attention_output)

        outputs = (layer_out,) + outputs
        return outputs

    def feed_forward(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class AntiLayerSegment(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config
        self.attention_typ = config.attention_type
        self.attention = AntiAttentionSegment(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise TypeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = AntiAttentionSegment(config)
        self.intermediate = AntiIntermediate(config)
        self.output = AntiOutput(config)
    
    def forward(self,
            hidden_states: torch.Tensor,
            x:torch.Tensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
                ):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            x,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        
        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )
          
            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                x,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_out = self.feed_forward(attention_output)

        outputs = (layer_out,) + outputs
        return outputs

    def feed_forward(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
class Structure_layer(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config 
        self.convert = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size,bias=False),
                                     nn.ReLU(inplace=True))
    def forward(self,x:torch.Tensor):
        return self.convert(x)
    
class AntiEncoder(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config
        if config.segment:
            self.layer = nn.ModuleList([AntiLayerSegment(config) for _ in range(config.num_hidden_layers)])
        else:
            self.layer = nn.ModuleList([AntiLayer(config) for _ in range(config.num_hidden_layers)])
        if config.structure:
            self.structure_layer = nn.ModuleList([Structure_layer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
    def forward(
        self,
        hidden_states: torch.Tensor,
        x:torch.Tensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        structure: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:

                if use_cache:
                    print(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward
                if self.config.segment:

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    x,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    )
                else:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                    )
            else:
                if self.config.segment:
                    layer_outputs = layer_module(
                        hidden_states,
                        x,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )
            
            hidden_states = layer_outputs[0]
            (attention_mask,) = adjust_tensors_for_parallel(hidden_states, attention_mask)

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class AntiPooler(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states:torch.Tensor) -> torch.Tensor:
        # 'pool' the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:,0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class AntiPredictionHeadTransform(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class AntiPredictNextResidues(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transfrom = AntiPredictionHeadTransform(config)

        self.decoder = nn.Linear(config.hidden_size,config.residue_size)

        self.bias = nn.Parameter(torch.zeros(config.residue_size))

        self.decoder.bias = self.bias

    def forward(self, hidden_states, inv_lang_adapter=None):
        hidden_states = self.transfrom(hidden_states)
        if inv_lang_adapter:
            hidden_states = inv_lang_adapter(hidden_states, rev=True)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class AntiModelIinitial():
    def __init__(self,config) -> None:
        self.config = config
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, AntiEncoder):
            module.gradient_checkpointing = value




if __name__ == '__main__':
    config = configuration()
    # EM = AntiEmbeddings(config)
    # setattr(config,'one_hot',True)
    # x1 = torch.arange(12).repeat_interleave(3).reshape(3,-1)
    # x1 = F.one_hot(x1,config.token_size).float()
    
    # x2 = torch.arange(3).repeat_interleave(12).reshape(3,-1)
    # x3 = torch.arange(12).repeat_interleave(3).reshape(3,-1)
    # example = EM(x1,x2,x3)
    # pdb.set_trace()
    # attention = AntiSelfAttention(config)
    # inter = AntiIntermediate(config)
    # out = AntiSelfOutput(config)
    # q = torch.rand([3,12,768])
    # k = torch.rand([3,24,768])
    # v = torch.rand([3,24,768])
    # x = attention.forward(hidden_states=q,encoder_hidden_states=k)
    
    # antiattention = AntiAttention(config=config)
    # x = antiattention.forward(hidden_states=q)
    
    # antilayer = AntiLayer(config)
    # x = antilayer.forward(hidden_states=q)
    # antiencoder = AntiEncoder(config)
    # x = antiencoder.forward(hidden_states=q)
    import time
    setattr(config,'segment',True)
    setattr(config,'max_position_embeddings',1024)
    setattr(config,'segments',4)
    setattr(config,'segment_size',int(config.max_position_embeddings/config.segments))
    setattr(config,'max_position_embeddings',1024+config.segments+1)
    s = time.time()
    for i in range(10):
        attention_segment = AntiAttentionSegment(config)
        x = torch.rand((5,1024+5,768))
        y = attention_segment(x)
        
        sum(sum(sum(y[0]))).backward()
    e = time.time()
    print("segment",e-s)
    

    
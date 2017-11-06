
# coding: utf-8

# # DNCore
# 
# #### Hybrid computing using a neural network with dynamic external memory
# #### differentiable neural computer
# 
# ##### 2017-09-02 DNCore封装
# - calculate 该模块不参与时序间传递
# - update 该模块参与时序间传递，需要上一时刻模块状态

# In[ ]:

import numpy as np
import tensorflow as tf

from sonnet.python.modules.base import AbstractModule
from sonnet.python.modules.basic import BatchApply, Linear, BatchFlatten
from sonnet.python.modules.rnn_core import RNNCore
from sonnet.python.modules.gated_rnn import LSTM
from sonnet.python.modules.basic_rnn import DeepRNN


# ## Access
# ### Memory Addressing 

# #### 基于内容寻址机制 Content-based addressing
# - 使用余弦相似性处理外存储器Access 记忆矩阵中数值相似性
# - 读头控制和写头控制寻址机制组成模块

# In[ ]:

# Content-based addressing
class calculate_Content_based_addressing(AbstractModule):
    """
    查询计算记忆矩阵每行内容记忆之间的余弦相似度，
    使用softmax返回一个数值大小嵌入[0,1]区间tensor。
    
    """
    
    def __init__(self,
                num_heads, 
                word_size,
                epsilon = 1e-6,
                name='content_based_addressing'):
        """
        Initializes the module.

        Args:
          num_heads: number of memory write heads or read heads.
          word_size: memory word size.
          epsilon: 鲁棒性功能添加
          name: module name (default 'content_based_addressing')
        """
        super().__init__(name=name) # 调用父类初始化
        self._num_heads = num_heads
        self._word_size = word_size
        self._epsilon = epsilon
        

    def _clip_L2_norm(self, tensor, axis=2):
        """
        计算L2范数，余弦相似度公式分母，
        这里进行数值平稳化处理
        memory: A 3-D tensor of shape [batch_size, memory_size, word_size]
        keys: A 3-D tensor of shape [batch_size, num_heads, word_size]  
        """
        quadratic_sum = tf.reduce_sum(tf.multiply(tensor, tensor), axis=axis, keep_dims=True)    
        return tf.sqrt(quadratic_sum + self._epsilon)
    
    
    def _calculate_cosine_similarity(self, keys, memory):
        
        """
        计算余弦相似度
        Args:      
            memory: A 3-D tensor of shape [batch_size, memory_size, word_size]
            keys: A 3-D tensor of shape [batch_size, num_heads, word_size]  
        Returns:
            cosine_similarity: A 3-D tensor of shape `[batch_size, num_heads, memory_size]`.
        """
        # 分子
        matmul = tf.matmul(keys, memory, adjoint_b=True)
        # 分母
        memory_norm = self._clip_L2_norm(memory, axis=2)
        keys_norm = self._clip_L2_norm(keys, axis=2)
        # 余弦相似度计算， 添加epsilon消除极值影响；
        cosine_similarity = matmul / (tf.matmul(keys_norm, memory_norm, adjoint_b=True) + self._epsilon)
        return cosine_similarity
    
    
    def _build(self, memory, keys, strengths):
        """
        Connects the CosineWeights module into the graph.
        计算余弦相似度
        使用write strength或者read strength 适度缩放余弦相似度。
        提高不同读写头的读头控制、写头控制区分度。

        Args:
            memory: A 3-D tensor of shape `[batch_size, memory_size, word_size]`.
            keys: A 3-D tensor of shape `[batch_size, num_heads, word_size]`.
            strengths: A 2-D tensor of shape `[batch_size, num_heads]`.

        Returns:
            cosine_similarity: A 3-D tensor of shape `[batch_size, num_heads, memory_size]`.
            content_weighting: Weights tensor of shape `[batch_size, num_heads, memory_size]`.
        """    
        cosine_similarity = self._calculate_cosine_similarity(keys=keys, memory=memory)
        transformed_strengths = tf.expand_dims(strengths, axis=-1)
        sharp_activations = cosine_similarity * transformed_strengths
        softmax = BatchApply(module_or_op=tf.nn.softmax)
        return softmax(sharp_activations)


# #### 动态内存允许 Dynamic memory allocation
# - To allow the controller to free and allocate memory as needed, 
# - we developed a differentiable analogue of the ‘free list’ memory allocation scheme,
# - whereby a list of available memory locations is maintained 
# - by adding to and removing addresses from a linked list. 
# 

# In[ ]:

#Dynamic_memory_allocation
class update_Dynamic_memory_allocation(RNNCore):
    """
    Memory usage that is increased by writing and decreased by reading.

    This module is a pseudo-RNNCore whose state is a tensor with values in
    the range [0, 1] indicating the usage of each of `memory_size` memory slots.

    The usage is:

    *   Increased by writing, where usage is increased towards 1 at the write
      addresses.
    *   Decreased by reading, where usage is decreased after reading from a
      location when free_gates is close to 1.
    """  
    
    def __init__(
        self,
        memory_size,
        epsilon = 1e-6,
        name='dynamic_memory_allocation'):
        
        """Creates a module for dynamic memory allocation.

        Args:
          memory_size: Number of memory slots.
          name: Name of the module.
        """
        super().__init__(name=name)
        self._memory_size = memory_size
        self._epsilon = epsilon
    
    
    def _build(
        self,
        prev_usage,
        prev_write_weightings,
        free_gates,
        prev_read_weightings,
        write_gates, 
        num_writes):
        
        # 更新记忆矩阵每行的使用程度，区间[0,1]
        # 程度数值随写入行为提高，读取行为降低
        usage = self._update_usage_vector(
            prev_usage, 
            prev_write_weightings,
            free_gates, 
            prev_read_weightings)
        
        # 记忆矩阵行位置释放
        allocation_weightings = self._update_allocation_weightings(
            usage, write_gates, num_writes)
        
        return usage, allocation_weightings
    
    
    def _update_usage_vector(
        self, 
        prev_usage,
        prev_write_weightings,
        free_gates, 
        prev_read_weightings):
        """
        The usage is:

        *   Increased by writing, where usage is increased towards 1 at the write
          addresses.
        *   Decreased by reading, where usage is decreased after reading from a
          location when free_gates is close to 1.
        
        Args:
            prev_usage: tensor of shape `[batch_size, memory_size]` giving
            usage u_{t - 1} at the previous time step, with entries in range [0, 1].
        
            prev_write_weightings: tensor of shape `[batch_size, num_writes, memory_size]` 
            giving write weights at previous time step.
            
            free_gates: tensor of shape `[batch_size, num_reads]` which indicates
            which read heads read memory that can now be freed.
          
            prev_read_weightings: tensor of shape `[batch_size, num_reads, memory_size]` 
            giving read weights at previous time step.
          
        Returns:
            usage: tensor of shape `[batch_size, memory_size]` representing updated memory usage.
        """
        prev_write_weightings = tf.stop_gradient(prev_write_weightings)
        usage = self._calculate_usage_vector(prev_usage, prev_write_weightings)
        retention = self._calculate_retention_vector(free_gates, prev_read_weightings)
        return usage * retention
    
    
    def _calculate_usage_vector(
        self, 
        prev_usage, 
        prev_write_weightings):
        """
        注意这里usage更新使用上一个时间步的数据
        这个函数是特别添加处理多个写头写头控制情况,
        这个函数计算在写头操作之后记忆矩阵的使用情况usage
        
        Calcualtes the new usage after writing to memory.

        Args:
          prev_usage: tensor of shape `[batch_size, memory_size]`.
          write_weightings: tensor of shape `[batch_size, num_writes, memory_size]`.

        Returns:
          New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        with tf.name_scope('usage_after_write'):
            # Calculate the aggregated effect of all write heads
            fit_prev_write_weightings =             1 - tf.reduce_prod(1 - prev_write_weightings, axis=[1])
            
            usage_without_free =             prev_usage + fit_prev_write_weightings - prev_usage * fit_prev_write_weightings
            
            return usage_without_free
        
        
    def _calculate_retention_vector(
        self, 
        free_gates, 
        prev_read_weightings):
        
        """
        The memory retention vector phi_t represents by how much each location 
        will not be freed by the gates.
        
        Args:
            free_gates: tensor of shape `[batch_size, num_reads]` with entries in the
            range [0, 1] indicating the amount that locations read from can be
            freed.
            
            prev_write_weightings: tensor of shape `[batch_size, num_writes, memory_size]`.
        Returns:
            retention vector: [batch_size, memory_size]
        """
        with tf.name_scope('usage_after_read'):
            free_gates = tf.expand_dims(free_gates, axis=-1)
            
            retention_vector = tf.reduce_prod(
                1 - free_gates * prev_read_weightings, 
                axis=[1], name='retention')
            return retention_vector     
        
        
    def _update_allocation_weightings(
        self, 
        usage, 
        write_gates, 
        num_writes):
        
        """
        Calculates freeness-based locations for writing to.

        This finds unused memory by ranking the memory locations by usage, for each
        write head. (For more than one write head, we use a "simulated new usage"
        which takes into account the fact that the previous write head will increase
        the usage in that area of the memory.)

        Args:
            usage: A tensor of shape `[batch_size, memory_size]` representing
            current memory usage.

            write_gates: A tensor of shape `[batch_size, num_writes]` with values in
            the range [0, 1] indicating how much each write head does writing
            based on the address returned here (and hence how much usage
            increases).

            num_writes: The number of write heads to calculate write weights for.

        Returns:
            tensor of shape `[batch_size, num_writes, memory_size]` containing the
            freeness-based write locations. Note that this isn't scaled by `write_gate`; 
            this scaling must be applied externally.
        """
        with tf.name_scope('update_allocation'):
            write_gates = tf.expand_dims(write_gates, axis=-1)
            allocation_weightings = []
            for i in range(num_writes):
                allocation_weightings.append(
                    self._calculate_allocation_weighting(usage))
                # update usage to take into account writing to this new allocation
                usage += ((1-usage) * write_gates[:,i,:] * allocation_weightings[i])
            return tf.stack(allocation_weightings, axis=1)
        
        
    def _calculate_allocation_weighting(self, usage):
        
        """
        Computes allocation by sorting `usage`.

        This corresponds to the value a = a_t[\phi_t[j]] in the paper.

        Args:
              usage: tensor of shape `[batch_size, memory_size]` indicating current
              memory usage. This is equal to u_t in the paper when we only have one
              write head, but for multiple write heads, one should update the usage
              while iterating through the write heads to take into account the
              allocation returned by this function.

        Returns:
          Tensor of shape `[batch_size, memory_size]` corresponding to allocation.
        """
        with tf.name_scope('allocation'):
            # Ensure values are not too small prior to cumprod.
            usage = self._epsilon + (1 - self._epsilon) * usage
            non_usage = 1 - usage
            
            sorted_non_usage, indices = tf.nn.top_k(
            non_usage, k = self._memory_size, name='sort')
            
            sorted_usage = 1 - sorted_non_usage
            prod_sorted_usage = tf.cumprod(sorted_usage, axis=1, exclusive=True)
            
            sorted_allocation_weighting = sorted_non_usage * prod_sorted_usage
            
            # This final line "unsorts" sorted_allocation, so that the indexing
            # corresponds to the original indexing of `usage`.
            inverse_indices = self._batch_invert_permutation(indices)
            allocation_weighting = self._batch_gather(
                sorted_allocation_weighting, inverse_indices)
            
            return allocation_weighting
            

    def _batch_invert_permutation(self, permutations):
        
        """
        Returns batched `tf.invert_permutation` for every row in `permutations`.
        """
        
        with tf.name_scope('batch_invert_permutation', values=[permutations]):
            unpacked = tf.unstack(permutations, axis=0)
            
            inverses = [tf.invert_permutation(permutation) for permutation in unpacked]
            return tf.stack(inverses, axis=0)
        
              
    def _batch_gather(self, values, indices):
        """Returns batched `tf.gather` for every row in the input."""
        
        with tf.name_scope('batch_gather', values=[values, indices]):
            unpacked = zip(tf.unstack(values), tf.unstack(indices))
            result = [tf.gather(value, index) for value, index in unpacked]
            return tf.stack(result)   
        
    @property
    def state_size(self):
        pass
    
    @property
    def output_size(self):
        pass


# #### 基于读写行为顺序头控制 Temporal memory linkage

# In[ ]:

#Temporal_memory_linkage
class update_Temporal_memory_linkage(RNNCore):
    """
    Keeps track of write order for forward and backward addressing.

    This is a pseudo-RNNCore module, whose state is a pair `(link,
    precedence_weights)`, where `link` is a (collection of) graphs for (possibly
    multiple) write heads (represented by a tensor with values in the range
    [0, 1]), and `precedence_weights` records the "previous write locations" used
    to build the link graphs.

    The function `directional_read_weights` computes addresses following the
    forward and backward directions in the link graphs.
    """ 
    def __init__(self, 
                 memory_size, 
                 num_writes, 
                 name='temporal_memory_linkage'):
        
        """
        Construct a TemporalLinkage module.

        Args:
          memory_size: The number of memory slots.
          num_writes: The number of write heads.
          name: Name of the module.
        """  
        super().__init__(name=name)
        self._memory_size = memory_size
        self._num_writes = num_writes
        
        
    def _build(self, 
               prev_link, 
               prev_precedence_weightings,
               prev_read_weightings,
               write_weightings):
        """
        calculate the updated linkage state given the write weights.
        
        Args:           
            prev_links: A tensor of shape `[batch_size, num_writes, memory_size, memory_size]` 
            representing the previous link graphs for each write head.

            prev_precedence_weightings: A tensor of shape `[batch_size, num_writes, memory_size]` 
            containing the previous precedence weights.

            write_weightings: A tensor of shape `[batch_size, num_writes, memory_size]`
            containing the memory addresses of the different write heads.
        
        Returns:
            link:  A tensor of shape `[batch_size, num_writes, memory_size, memory_size]` 
            precedence_weightings: A tensor of shape `[batch_size, num_writes, memory_size]` 

        """
        link = self._update_link_matrix(
            prev_link, prev_precedence_weightings, write_weightings)
        
        precedence_weightings =         self._update_precedence_weightings(
            prev_precedence_weightings, write_weightings)
        
        forward_weightings =         self._calculate_directional_read_weightings(
            link, prev_read_weightings, forward=True)
        
        backward_weightings =         self._calculate_directional_read_weightings(
            link, prev_read_weightings, forward=False)
        
        return link, precedence_weightings, forward_weightings, backward_weightings  
    
    
    def _update_link_matrix(self, 
                            prev_link, 
                            prev_precedence_weightings, 
                            write_weightings):
        """
        calculates the new link graphs.

        For each write head, the link is a directed graph (represented by a matrix
        with entries in range [0, 1]) whose vertices are the memory locations, and
        an edge indicates temporal ordering of writes.

        Args:
          prev_links: A tensor of shape `[batch_size, num_writes, memory_size, memory_size]` 
          representing the previous link graphs for each write head.

          prev_precedence_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
          which is the previous "aggregated" write weights for each write head.

          write_weightings: A tensor of shape `[batch_size, num_writes, memory_size]` 
              containing the new locations in memory written to.

        Returns:
          A tensor of shape `[batch_size, num_writes, memory_size, memory_size]`
          containing the new link graphs for each write head.
        """    
        
        with tf.name_scope('link'):
                   
            write_weightings_i = tf.expand_dims(write_weightings, axis=3)
            write_weightings_j = tf.expand_dims(write_weightings, axis=2)
            prev_link_scale = 1 - write_weightings_i - write_weightings_j
            remove_old_link = prev_link_scale * prev_link
            
            prev_precedence_weightings_j = tf.expand_dims(
                prev_precedence_weightings, axis=2)
            add_new_link = write_weightings_i * prev_precedence_weightings_j
            
            link = remove_old_link + add_new_link
            
            #Return the link with the diagonal set to zero, to remove self-looping edges.
            batch_size = prev_link.get_shape()[0].value
            mask = tf.zeros(shape=[batch_size, self._num_writes, self._memory_size], 
                            dtype=prev_link.dtype)
            
            fit_link = tf.matrix_set_diag(link, diagonal=mask)
            return fit_link
        
        
    def _update_precedence_weightings(self, 
                                     prev_precedence_weightings, 
                                     write_weightings):
        """
        calculates the new precedence weights given the current write weights.

        The precedence weights are the "aggregated write weights" for each write
        head, where write weights with sum close to zero will leave the precedence
        weights unchanged, but with sum close to one will replace the precedence
        weights.   

        Args:
          prev_precedence_weightings: A tensor of shape `[batch_size, num_writes, memory_size]` 
          containing the previous precedence weights.

          write_weightings: A tensor of shape `[batch_size, num_writes, memory_size]`
          containing the new write weights.

        Returns:
          A tensor of shape `[batch_size, num_writes, memory_size]` 
          containing the new precedence weights.  
        """
        with tf.name_scope('precedence_weightings'):
            sum_writing = tf.reduce_sum(write_weightings, axis=2, keep_dims=True)
            
            precedence_weightings =             (1 - sum_writing) * prev_precedence_weightings + write_weightings
            
            return precedence_weightings
        

    def _calculate_directional_read_weightings(self,
                                               link, 
                                               prev_read_weightings, 
                                               forward):
        """
        calculates the forward or the backward read weightings.

        For each read head (at a given address), there are `num_writes` link graphs to follow. 
        Thus this function computes a read address for each of the
        `num_reads * num_writes` pairs of read and write heads.

        Args:
            link: tensor of shape `[batch_size, num_writes, memory_size, memory_size]`
            representing the link graphs L_t.

            prev_read_weightsing: tensor of shape `[batch_size, num_reads, memory_size]` 
            containing the previous read weights w_{t-1}^r.

            forward: Boolean indicating whether to follow the "future" direction in 
            the link graph (True) or the "past" direction (False).

        Returns:
            tensor of shape `[batch_size, num_reads, num_writes, memory_size]`

            Note: We calculate the forward and backward directions for each pair of
            read and write heads; hence we need to tile the read weights and do a
            sort of "outer product" to get this.
        """
        with tf.name_scope('directional_read_weightings'):
            # We calculate the forward and backward directions for each pair of
            # read and write heads; hence we need to tile the read weights and do a
            # sort of "outer product" to get this.
            expanded_read_weightings =             tf.stack([prev_read_weightings] * self._num_writes, axis=1)
            directional_weightings = tf.matmul(expanded_read_weightings, link, adjoint_b=forward)
            # Swap dimensions 1, 2 so order is [batch, reads, writes, memory]:
            return tf.transpose(directional_weightings, perm=[0,2,1,3])    


# ### Access

# In[ ]:

# MemoryAccess
class MemoryAccess(RNNCore):
    """
    Access module of the Differentiable Neural Computer.

    This memory module supports multiple read and write heads. It makes use of:

    *   `update_Temporal_memory_linkage` to track the temporal 
    ordering of writes in memory for each write head.
    
    *   `update_Dynamic_memory_allocation` for keeping track of 
    memory usage, where usage increase when a memory location is 
    written to, and decreases when memory is read from that 
    the controller says can be freed.
      
    Write-address selection is done by an interpolation between content-based
    lookup and using unused memory.
    
    Read-address selection is done by an interpolation of content-based lookup
    and following the link graph in the forward or backwards read direction.
    """
    
    def __init__(self, 
                 memory_size = 128, 
                 word_size = 20, 
                 num_reads = 1, 
                 num_writes = 1, 
                 name='memory_access'):
        
        """
        Creates a MemoryAccess module.

        Args:
            memory_size: The number of memory slots (N in the DNC paper).
            word_size: The width of each memory slot (W in the DNC paper)
            num_reads: The number of read heads (R in the DNC paper).
            num_writes: The number of write heads (fixed at 1 in the paper).
            name: The name of the module.
        """
        super().__init__(name=name)
        self._memory_size = memory_size
        self._word_size = word_size
        self._num_reads = num_reads
        self._num_writes = num_writes
        
        self._write_content_mod = calculate_Content_based_addressing(
            num_heads = self._num_writes, 
            word_size = self._word_size, 
            name = 'write_content_based_addressing')
        
        self._read_content_mod = calculate_Content_based_addressing(
            num_heads = self._num_reads, 
            word_size = self._word_size, 
            name = 'read_content_based_addressing')
        
        self._temporal_linkage = update_Temporal_memory_linkage(
            memory_size = self._memory_size, 
            num_writes = self._num_writes)
        
        self._dynamic_allocation = update_Dynamic_memory_allocation(
            memory_size = self._memory_size)
        
        
    def _build(self, interface_vector, prev_state):
        """
        Connects the MemoryAccess module into the graph.

        Args:
            inputs: tensor of shape `[batch_size, input_size]`. 
            This is used to control this access module.
            
            prev_state: Instance of `AccessState` containing the previous state.

        Returns:
            A tuple `(output, next_state)`, where `output` is a tensor of shape
            `[batch_size, num_reads, word_size]`, and `next_state` is the new
            `AccessState` named tuple at the current time t.
        """
        tape = self._Calculate_interface_parameters(interface_vector)
        
        prev_memory,        prev_read_weightings,        prev_write_weightings,        prev_precedence_weightings,        prev_link,        prev_usage = prev_state

        # 更新写头
        write_weightings,        usage =         self._update_write_weightings(tape, 
                                      prev_memory, 
                                      prev_usage, 
                                      prev_write_weightings, 
                                      prev_read_weightings)
        
        # 更新记忆
        memory = self._update_memory(prev_memory, 
                                     write_weightings, 
                                     tape['erase_vectors'], 
                                     tape['write_vectors'])
        
        # 更新读头
        read_weightings,        link,        precedence_weightings=         self._update_read_weightings(tape, 
                                     memory, 
                                     write_weightings,
                                     prev_read_weightings, 
                                     prev_precedence_weightings,
                                     prev_link)

        read_vectors = tf.matmul(read_weightings, memory)
        
        state = (memory,
                 read_weightings,
                 write_weightings,
                 precedence_weightings,
                 link,
                 usage)
        
        return read_vectors, state
        

    def _update_write_weightings(self, 
                                  tape, 
                                  prev_memory, 
                                  prev_usage, 
                                  prev_write_weightings, 
                                  prev_read_weightings):       
        """
        Calculates the memory locations to write to.

        This uses a combination of content-based lookup and finding an unused
        location in memory, for each write head.

        Args:
            tape: Collection of inputs to the access module, including controls for
            how to chose memory writing, such as the content to look-up and the
            weighting between content-based and allocation-based addressing.
            
            memory: A tensor of shape  `[batch_size, memory_size, word_size]`
            containing the current memory contents.
            
            usage: Current memory usage, which is a tensor of shape 
            `[batch_size, memory_size]`, used for allocation-based addressing.

        Returns:
            tensor of shape `[batch_size, num_writes, memory_size]` 
            indicating where to write to (if anywhere) for each write head.
        """
        with tf.name_scope('update_write_weightings',                            values=[tape, prev_memory, prev_usage]):
            
            write_content_weightings =             self._write_content_mod(
                prev_memory, 
                tape['write_content_keys'], 
                tape['write_content_strengths'])
            
            usage, write_allocation_weightings =             self._dynamic_allocation(
                prev_usage, 
                prev_write_weightings, 
                tape['free_gates'], 
                prev_read_weightings, 
                tape['write_gates'], 
                self._num_writes)
            
            allocation_gates = tf.expand_dims(tape['allocation_gates'], axis=-1)
            write_gates = tf.expand_dims(tape['write_gates'], axis=-1)
            
            write_weightings = write_gates *             (allocation_gates * write_allocation_weightings +              (1 - allocation_gates) * write_content_weightings)
            
            return write_weightings, usage
        
        
    def _update_memory(self, 
                       prev_memory, 
                       write_weightings, 
                       erase_vectors, 
                       write_vectors):
        """
        Args:
            prev_memory: 3-D tensor of shape `[batch_size, memory_size, word_size]`.
            write_weightings: 3-D tensor `[batch_size, num_writes, memory_size]`.
            erase_vectors: 3-D tensor `[batch_size, num_writes, word_size]`.
            write_vectors: 3-D tensor `[batch_size, num_writes, word_size]`.

      Returns:
            memory: 3-D tensor of shape `[batch_size, num_writes, word_size]`.
        """
        with tf.name_scope('erase_old_memory',                            values=[prev_memory, 
                                   write_weightings, 
                                   erase_vectors]):

            expand_write_weightings =             tf.expand_dims(write_weightings, axis=3)

            expand_erase_vectors =             tf.expand_dims(erase_vectors, axis=2)

            # 这里有多个写头，需要使用累成处理多个写头
            erase_gates =             expand_write_weightings * expand_erase_vectors

            retention_gate =             tf.reduce_prod(1 - erase_gates, axis=[1])

            retention_memory = prev_memory * retention_gate

        with tf.name_scope('additive_new_memory',                            values=[retention_memory, 
                                   write_weightings, 
                                   write_vectors]):

            memory = retention_memory +             tf.matmul(write_weightings, write_vectors, adjoint_a=True)

            return memory
        
    
    def _Calculate_interface_parameters(self, interface_vector):
        """
        Interface parameters. 
        Before being used to parameterize the memory interactions, 
        the individual components are then processed with various 
        functions to ensure that they lie in the correct domain.     
        """
        # read_keys: [batch_size, num_reads, word_size]
        read_keys = Linear(
            output_size= self._num_reads * self._word_size,
            name='read_keys')(interface_vector)
        read_keys = tf.reshape(
            read_keys, shape=[-1, self._num_reads, self._word_size])

        # write_keys: [batch_size, num_writes, word_size]
        write_keys = Linear(
            output_size= self._num_writes * self._word_size, 
            name= 'write_keys')(interface_vector)
        write_keys = tf.reshape(
            write_keys, shape=[-1, self._num_writes, self._word_size])


        # read_strengths: [batch_size, num_reads]
        read_strengths = Linear(
            output_size= self._num_reads,
            name= 'read_strengths')(interface_vector)
        read_strengths = 1 + tf.nn.softplus(read_strengths)

        # write_strengths: [batch_size, num_writes]
        write_strengths = Linear(
            output_size= self._num_writes,
            name='write_strengths')(interface_vector)
        write_strengths = 1 + tf.nn.softplus(write_strengths)


        # earse_vector: [batch_size, num_writes * word_size]
        erase_vectors = Linear(
            output_size= self._num_writes * self._word_size,
            name='erase_vectors')(interface_vector)
        erase_vectors = tf.reshape(
            erase_vectors, shape=[-1, self._num_writes, self._word_size])
        erase_vectors = tf.nn.sigmoid(erase_vectors)

        # write_vectors: [batch_size, num_writes * word_size]
        write_vectors = Linear(
            output_size= self._num_writes * self._word_size,
            name='write_vectors')(interface_vector)
        write_vectors = tf.reshape(
            write_vectors, shape=[-1, self._num_writes, self._word_size])


        # free_gates: [batch_size, num_reads]
        free_gates = Linear(
            output_size= self._num_reads,
            name='free_gates')(interface_vector)
        free_gates = tf.nn.sigmoid(free_gates)

        # allocation_gates: [batch_size, num_writes]
        allocation_gates = Linear(
            output_size= self._num_writes,
            name='allocation_gates')(interface_vector)
        allocation_gates = tf.nn.sigmoid(allocation_gates)

        # write_gates: [batch_size, num_writes]
        write_gates = Linear(
            output_size= self._num_writes,
            name='write_gates')(interface_vector)
        write_gates = tf.nn.sigmoid(write_gates)

        # read_modes: [batch_size, (1 + 2 * num_writes) * num_reads]
        num_read_modes = 1 + 2 * self._num_writes
        read_modes = Linear(
            output_size= self._num_reads * num_read_modes,
            name='read_modes')(interface_vector)
        read_modes = tf.reshape(
            read_modes, shape=[-1, self._num_reads, num_read_modes])
        read_modes = BatchApply(tf.nn.softmax)(read_modes)

        tape = {
            'read_content_keys': read_keys,
            'read_content_strengths': read_strengths,
            'write_content_keys': write_keys,
            'write_content_strengths': write_strengths,
            'write_vectors': write_vectors,
            'erase_vectors': erase_vectors,
            'free_gates': free_gates,
            'allocation_gates': allocation_gates,
            'write_gates': write_gates,
            'read_modes': read_modes,
        }
        return tape        


    def _update_read_weightings(self, 
                                tape, 
                                memory, 
                                write_weightings,
                                prev_read_weightings, 
                                prev_precedence_weightings, 
                                prev_link):
        """
        Calculates read weights for each read head.

        The read weights are a combination of following the link graphs in the
        forward or backward directions from the previous read position, and doing
        content-based lookup. The interpolation between these different modes is
        done by `inputs['read_mode']`.

        Args:
            inputs: Controls for this access module. 
            This contains the content-based keys to lookup, 
            and the weightings for the different read modes.

            memory: A tensor of shape `[batch_size, memory_size, word_size]`
            containing the current memory contents to do content-based lookup.

            prev_read_weights: A tensor of shape `[batch_size, num_reads, memory_size]` 
            containing the previous read locations.

            link: A tensor of shape `[batch_size, num_writes, memory_size, memory_size]` 
            containing the temporal write transition graphs.

        Returns:
            A tensor of shape `[batch_size, num_reads, memory_size]` 
            containing the read weights for each read head.
        """    
        with tf.name_scope(
            'update_read_weightings', 
            values=[tape, 
                    memory, 
                    prev_read_weightings, 
                    prev_precedence_weightings, 
                    prev_link]):

            read_content_weightings =             self._read_content_mod(
                memory, 
                tape['read_content_keys'], 
                tape['read_content_strengths'])

            
            link,            precedence_weightings,            forward_weightings,            backward_weightings =             self._temporal_linkage(
                prev_link, 
                prev_precedence_weightings, 
                prev_read_weightings,
                write_weightings)
            
            
            backward_mode = tape['read_modes'][:, :, :self._num_writes]
            forward_mode = tape['read_modes'][:, :, self._num_writes:2 * self._num_writes]
            content_mode = tape['read_modes'][:, :, 2 * self._num_writes]
            
            backward_ = tf.expand_dims(backward_mode, axis=3) * backward_weightings
            backward_ = tf.reduce_sum(backward_, axis=2)
            
            forward_ = tf.expand_dims(forward_mode, axis=3) * forward_weightings
            forward_ = tf.reduce_sum(forward_, axis=2)
            
            content_ = tf.expand_dims(content_mode, axis=2) * read_content_weightings

            read_weightings = backward_ + forward_ + content_

            return read_weightings, link, precedence_weightings
        
    
    @property
    def state_size(self):
        """Returns a tuple of the shape of the state tensors."""
        memory = tf.TensorShape([self._memory_size, self._word_size])
        read_weightings = tf.TensorShape([self._num_reads, self._memory_size])
        write_weightings = tf.TensorShape([self._num_writes, self._memory_size])
        link = tf.TensorShape([self._num_writes, self._memory_size, self._memory_size])
        precedence_weightings = tf.TensorShape([self._num_writes, self._memory_size])
        usage = tf.TensorShape([self._memory_size])
        return (memory, 
                read_weightings, 
                write_weightings,
                precedence_weightings,
                link, 
                usage)
    
    
    @property
    def output_size(self):
        """
        Returns the output shape.
        """
        return tf.TensorShape([self._num_reads, self._word_size])


# # DNCore 封装

# #### LSTM

# In[ ]:

class DNCoreLSTM(RNNCore):
    """
    单层LSTM控制器DNCore
    """
    
    def __init__(
        self,
        dnc_output_size,
        hidden_size= 128,
        forget_bias=1.0,
        initializers=None,
        partitioners=None,
        regularizers=None,
        use_peepholes=False,
        use_layer_norm=False,
        hidden_clip_value=None,
        cell_clip_value=None,
        custom_getter=None,
        memory_size= 256,
        word_size= 128, 
        num_read_heads= 3,
        num_write_heads= 1,
        name='DNCoreLSTM'):
        
        super().__init__(name=name) # 调用父类初始化
        with self._enter_variable_scope():
            controller = LSTM(
                hidden_size=hidden_size,
                forget_bias=forget_bias,
                initializers=initializers,
                partitioners=partitioners,
                regularizers=regularizers,
                use_peepholes=use_peepholes,
                use_layer_norm=use_layer_norm,
                hidden_clip_value=hidden_clip_value,
                cell_clip_value=cell_clip_value,
                custom_getter=custom_getter)
            
            self._controller = controller
            self._access = MemoryAccess(
                memory_size= memory_size, 
                word_size= word_size, 
                num_reads= num_read_heads, 
                num_writes= num_write_heads)  
            
            
        self._dnc_output_size = dnc_output_size
        self._num_read_heads = num_read_heads
        self._word_size = word_size
        
        
    def _build(self, inputs, prev_tape):
        
        prev_controller_state,        prev_access_state,        prev_read_vectors = prev_tape
        
        batch_flatten = BatchFlatten()
        controller_input = tf.concat(
            [batch_flatten(inputs), batch_flatten(prev_read_vectors)], axis= 1)
        
        # 控制器处理数据
        controller_output, controller_state =         self._controller(controller_input, prev_controller_state)
        
        # 外存储器交互
        read_vectors, access_state =         self._access(controller_output, prev_access_state)
        
        # DNC 输出
        dnc_output = tf.concat(
            [controller_output, batch_flatten(read_vectors)], axis= 1)
        dnc_output = Linear(
            self._dnc_output_size, name='dnc_output')(dnc_output)
        
        return dnc_output, (controller_state, access_state, read_vectors)
    
    
    def initial_state(self, batch_size, dtype=tf.float32):
        controller_state= self._controller.initial_state(batch_size, dtype)
        access_state= self._access.initial_state(batch_size, dtype)
        read_vectors= tf.zeros([batch_size, self._num_read_heads, self._word_size], dtype=dtype)
        return (controller_state, access_state, read_vectors)
    
    
    @property
    def state_size(self):
        controller_state= self._controller.state_size
        access_state= self._access.state_size
        read_vectors= tf.TensorShape([self._num_read_heads, self._word_size])
        return (controller_state, access_state, read_vectors)
    
    
    @property
    def output_size(self):
        return tf.TensorShape([self._dnc_output_size])


# #### DeepRNN

# In[ ]:

class DNCoreDeepLSTM(RNNCore):
    
    def __init__(
        self,
        dnc_output_size,
        hidden_size= 128,
        forget_bias=1.0,
        initializers=None,
        partitioners=None,
        regularizers=None,
        use_peepholes=False,
        use_layer_norm=False,
        hidden_clip_value=None,
        cell_clip_value=None,
        custom_getter=None,
        memory_size= 256,
        word_size= 128, 
        num_read_heads= 3,
        num_write_heads= 1,
        name='DNCoreDeepLSTM'):
        
        super().__init__(name=name) # 调用父类初始化
        with self._enter_variable_scope():
            
            layer_1 = LSTM(
                hidden_size=hidden_size,
                forget_bias=forget_bias,
                initializers=initializers,
                partitioners=partitioners,
                regularizers=regularizers,
                use_peepholes=use_peepholes,
                use_layer_norm=use_layer_norm,
                hidden_clip_value=hidden_clip_value,
                cell_clip_value=cell_clip_value,
                custom_getter=custom_getter)
            
            layer_2 = LSTM(
                hidden_size=hidden_size,
                forget_bias=forget_bias,
                initializers=initializers,
                partitioners=partitioners,
                regularizers=regularizers,
                use_peepholes=use_peepholes,
                use_layer_norm=use_layer_norm,
                hidden_clip_value=hidden_clip_value,
                cell_clip_value=cell_clip_value,
                custom_getter=custom_getter)
            
            layer_3 = LSTM(
                hidden_size=hidden_size,
                forget_bias=forget_bias,
                initializers=initializers,
                partitioners=partitioners,
                regularizers=regularizers,
                use_peepholes=use_peepholes,
                use_layer_norm=use_layer_norm,
                hidden_clip_value=hidden_clip_value,
                cell_clip_value=cell_clip_value,
                custom_getter=custom_getter)

            self._controller = DeepRNN([layer_1, layer_2, layer_3]) 
            self._access = MemoryAccess(
                memory_size= memory_size, 
                word_size= word_size, 
                num_reads= num_read_heads, 
                num_writes= num_write_heads)
            
        self._dnc_output_size = dnc_output_size
        self._num_read_heads = num_read_heads
        self._word_size = word_size
        
        
    def _build(self, inputs, prev_tape):
        
        prev_controller_state,        prev_access_state,        prev_read_vectors = prev_tape
        
        batch_flatten = BatchFlatten()
        controller_input = tf.concat(
            [batch_flatten(inputs), batch_flatten(prev_read_vectors)], axis= 1)
        
        # 控制器处理数据
        controller_output, controller_state =         self._controller(controller_input, prev_controller_state)
        
        # 外存储器交互
        read_vectors, access_state =         self._access(controller_output, prev_access_state)
        
        # DNC 输出
        dnc_output = tf.concat(
            [controller_output, batch_flatten(read_vectors)], axis= 1)
        dnc_output = Linear(
            self._dnc_output_size, name='dnc_output')(dnc_output)
        
        return dnc_output, (controller_state, access_state, read_vectors)
    
    
    def initial_state(self, batch_size, dtype=tf.float32):
        controller_state= self._controller.initial_state(batch_size, dtype)
        access_state= self._access.initial_state(batch_size, dtype)
        read_vectors= tf.zeros([batch_size, self._num_read_heads, self._word_size], dtype=dtype)
        return (controller_state, access_state, read_vectors)
    
    
    @property
    def state_size(self):
        controller_state= self._controller.state_size
        access_state= self._access.state_size
        read_vectors= tf.TensorShape([self._num_read_heads, self._word_size])
        return (controller_state, access_state, read_vectors)
    
    
    @property
    def output_size(self):
        return tf.TensorShape([self._dnc_output_size])


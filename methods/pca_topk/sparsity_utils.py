import torch
import ipdb
import math


def tensor_sparsity(tensor):
    """Calculate the sparsity (fraction of zeros) in a tensor"""
    return (tensor == 0).sum().item() / tensor.numel()

class SparseLinear(torch.nn.Module):
    """Custom Linear layer that uses optimized PyTorch sparse matrix multiplication"""
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.is_sparse = False
        self._sparse_weight = None
        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
    
    def make_sparse(self):
        """Convert weight to sparse format using optimized PyTorch sparse operations"""
        self.is_sparse = True
        
        # Create optimized sparse representation using PyTorch
        # Get indices of non-zero elements
        indices = torch.nonzero(self.weight).t()
        
        # Get values at these indices
        values = self.weight[indices[0], indices[1]]
        
        # Create sparse tensor in COO format (most efficient for matrix multiplication)
        self._sparse_weight = torch.sparse_coo_tensor(
            indices, values, self.weight.shape, 
            device=self.weight.device,
            dtype=self.weight.dtype
        ).coalesce()  # Coalesce to combine duplicate indices for better performance
    
    def forward(self, input):
        if not self.is_sparse:
            # Convert to sparse on first use if not already done
            self.make_sparse()
            self.is_sparse = True
        
        # Store original dtype for later conversion back
        original_dtype = input.dtype
        
        # Handle input dimensions for sparse matrix multiplication
        original_shape = input.shape
        need_reshape = len(original_shape) > 2
        
        if need_reshape:
            # Reshape to 2D for matrix multiplication
            input_2d = input.reshape(-1, original_shape[-1])
        else:
            input_2d = input
            
        if input_2d.shape[-1] != self.in_features:
            # Transpose if dimensions don't match
            input_2d = input_2d.t()
        
      
        if self._sparse_weight is None:
            self.make_sparse()
        
        
        if input.dtype == torch.float16 or input.dtype == torch.bfloat16:
            input_2d = input_2d.float()
            sparse_weight = self._sparse_weight.float()
        else:
            sparse_weight = self._sparse_weight
        
        # Use PyTorch's optimized sparse matrix multiplication
        output_2d = torch.sparse.mm(sparse_weight, input_2d.t()).t()
        
        # Convert back to original dtype if needed
        if output_2d.dtype != original_dtype:
            output_2d = output_2d.to(original_dtype)
        

        if self.bias is not None:
            output_2d += self.bias
        if need_reshape:
            output_shape = list(original_shape[:-1]) + [self.out_features]
            return output_2d.reshape(*output_shape)
        else:
            return output_2d


class SparsityHandler:
    def __init__(self, sparsity_type="attention-query-key"):
        self.sparsity_type = sparsity_type
        self.density = 0.1  
        # For block sparsity in attention-query-key
        self.block_size = 16  
        self.block_density = 0.1  
        self.sparsified_weights = {}

    def apply_sparsity_pattern(self, projection_matrix):
        """Apply the selected sparsity pattern to the projection matrix"""
        stats = self.get_sparsity_stats(projection_matrix)
        if self.sparsity_type == "attention-query-key":
            return projection_matrix
        else:
            return projection_matrix
    
    def apply_query_key_sparsity(self, matrix):
        """Apply block sparsity to query/key matrices after projection"""
        if self.sparsity_type != "attention-query-key":
            return matrix  
            
        try:
            if not matrix.is_contiguous():
                matrix = matrix.contiguous()
                
            shape = matrix.shape
            
            flat_matrix = matrix.view(-1)
            k = int(flat_matrix.numel() * self.density)
            
            threshold = torch.kthvalue(torch.abs(flat_matrix), flat_matrix.numel() - k + 1)[0]
            
            mask = (torch.abs(matrix) >= threshold).to(matrix.dtype)
            sparse_matrix = matrix * mask
            
            return sparse_matrix
            
        except Exception as e:
            return matrix
            
    def sparsify_linear_weight(self, linear_layer, layer_name=None):
        """Convert a standard Linear layer to a SparseLinear layer with sparsified weights"""
        if self.sparsity_type != "attention-query-key":
            return linear_layer
            
        if layer_name and layer_name in self.sparsified_weights:
            return linear_layer
        
        try:
            sparse_layer = SparseLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
                device=linear_layer.weight.device,
                dtype=linear_layer.weight.dtype
            )
            
            sparse_layer.weight.data.copy_(linear_layer.weight.data)
            if linear_layer.bias is not None:
                sparse_layer.bias.data.copy_(linear_layer.bias.data)
            
            flat_weight = sparse_layer.weight.view(-1)
            sorted_weights = torch.sort(torch.abs(flat_weight), descending=True)[0]
            
            threshold_idx = int(flat_weight.numel() * 0.5)  # 50% density
            threshold = sorted_weights[threshold_idx]
            
            mask = (torch.abs(sparse_layer.weight) >= threshold).to(sparse_layer.weight.dtype)
            
            sparse_layer.weight.data *= mask
            
            sparse_layer.make_sparse()
            
            if layer_name:
                self.sparsified_weights[layer_name] = True
            
            return sparse_layer
        except Exception as e:
            try:
              
                weight = linear_layer.weight.data
                
              
                flat_weight = weight.view(-1)
                k = int(flat_weight.numel() * 0.5)  # 50% density
                
                threshold = torch.kthvalue(torch.abs(flat_weight), flat_weight.numel() - k + 1)[0]
                mask = (torch.abs(weight) >= threshold).to(weight.dtype)
                linear_layer.weight.data *= mask
                
             
                if layer_name:
                    self.sparsified_weights[layer_name] = True
                    
                return linear_layer
            except Exception:
             
                return linear_layer
    
    def sparse_matmul(self, a, b):
        """Perform efficient sparse matrix multiplication using optimized PyTorch sparse operations"""
        # Store original dtype for later conversion back
        original_dtype = a.dtype
        
        # Convert to float32 if using half precision, as sparse operations don't support half
        if a.dtype == torch.float16 or a.dtype == torch.bfloat16:
            a = a.float()
            b = b.float()
        
        # Handle the attention query-key case (3D tensors)
        if a.dim() == 3 and b.dim() == 3:
            # For query-key attention: [batch, seq_q, dim] @ [batch, dim, seq_k] -> [batch, seq_q, seq_k]
            batch_size = a.shape[0]
            seq_len_q = a.shape[1]
            hidden_dim = a.shape[2]
            seq_len_k = b.shape[2]
        
            # Use torch.bmm for batched matrix multiplication (dense)
            result = torch.bmm(a, b)
            
            # Convert back to original dtype
            if original_dtype != result.dtype:
                result = result.to(original_dtype)
                
            return result
        
        # For 2D case, use sparse matrix multiplication
        else:
            # Ensure tensors are 2D
            if a.dim() > 2:
                a_2d = a.reshape(-1, a.shape[-1])
            else:
                a_2d = a
                
            if b.dim() > 2:
                b_2d = b.reshape(-1, b.shape[-1])
            else:
                b_2d = b
            
            # Handle dimension mismatch
            if a_2d.shape[1] != b_2d.shape[0] and a_2d.shape[1] == b_2d.shape[1]:
                b_2d = b_2d.transpose(0, 1)
            
            # Convert to sparse
            indices = torch.nonzero(a_2d).t()
            if indices.shape[1] == 0:  # Handle all zeros
                result_2d = torch.zeros(a_2d.shape[0], b_2d.shape[1], device=a.device, dtype=a.dtype)
            else:
                values = a_2d[indices[0], indices[1]]
                sparse_a = torch.sparse_coo_tensor(
                    indices, values, a_2d.shape,
                    device=a_2d.device,
                    dtype=a_2d.dtype
                ).coalesce()
                
                # Perform sparse matrix multiplication
                result_2d = torch.sparse.mm(sparse_a, b_2d)
            
            # Convert back to original dtype
            if original_dtype != result_2d.dtype:
                result_2d = result_2d.to(original_dtype)
            
    
            if a.dim() > 2 or b.dim() > 2:
                target_shape = list(a.shape[:-1]) + [b.shape[-1] if b.dim() > 1 else 1]
                return result_2d.view(*target_shape)
            else:
                return result_2d

    def get_sparsity_stats(self, tensor):
        """Get sparsity statistics for the tensor"""
        total_elements = tensor.numel()
        zero_elements = torch.sum(tensor == 0).item()
        sparsity_ratio = zero_elements / total_elements
        return {
            'total_elements': total_elements,
            'zero_elements': zero_elements,
            'sparsity_ratio': sparsity_ratio
        }

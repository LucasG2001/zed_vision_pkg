import torch
import time

start = time.time()
for i in range (100):
    batch_size = 10
    height = 1280
    width = 720

    # Number of 1's you want in the tensor
    num_ones = 500

    # Create a tensor filled with zeros
    tensor_dense = torch.zeros(batch_size, height, width)

    # Generate random indices
    random_indices = torch.randint(0, height * width, size=(batch_size, num_ones))
    # Flatten the tensor and set the selected indices to 1
    tensor_dense.view(batch_size, -1).scatter_(1, random_indices, 1)
    #print("tensor dense shape is ", tensor_dense.shape)
    torch.Size([10, 1280, 720])
    tensor_sparse = tensor_dense.to_sparse()
    #print("sparse tensor indices is shape ", tensor_sparse.indices().shape)
    #print("sanity check: sum of tensor is ", tensor_sparse.sum())
    indices = tensor_sparse.indices()
    values = tensor_sparse.values()
    fx, fy, fz = 2.0, 1.5, 0.8
    # Apply the transformations to the coordinates
    transformed_indices = torch.zeros_like(indices, dtype=torch.float32)
    transformed_indices[1] = indices[1].float() / fx
    transformed_indices[2] = indices[2].float() / fy
    transformed_values = values * fz
    # Apply the transformations to the coordinates
    #print("transformed indices have shape ", transformed_indices.shape)
    print("transformed values have shape ", transformed_values.shape)
    # Create a new sparse tensor with the transformed coordinates and values
    transformed_sparse_tensor = torch.sparse_coo_tensor(transformed_indices, transformed_values, tensor_dense.size())
    # Create a tensor for the shape (batch_size, 1280*720, 3)
    shape = (tensor_dense.size(0), 1280 * 720, 3)
    # Initialize a sparse tensor with zeros
    scaled_sparse_tensor = torch.sparse_coo_tensor(*shape)
    print("scaled sparse tensor is of shape ", scaled_sparse_tensor.shape)
    # Map the scaled coordinates and values to the sparse tensor
    # Set the indices[1], indices[2], and scaled_values in the last three dimensions
    scaled_sparse_tensor.indices()[0] = transformed_indices[0]
    scaled_sparse_tensor.indices()[1] = transformed_indices[1]
    scaled_sparse_tensor.indices()[2] = transformed_indices[2]
    # Set the values in the last dimension
    scaled_sparse_tensor.values()[:transformed_indices.size(1)] = transformed_values
    print("transformed_sparse_tensor is of shape ", transformed_sparse_tensor.shape)

end = time.time()
print(f"whole script took {end-start} seconds and one iteration on average {(end-start)/100} seconds")

def create_sparse_pointcloud_tensor(masked_depth_dense, intrinsic, depth_scale):
    print(f"input dense tensor has shape of ", masked_depth_dense.shape)
    cx = intrinsic.intrinsic_matrix[0, 2]
    cy = intrinsic.intrinsic_matrix[1, 2]
    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    # get sparse tensor
    tensor_sparse = masked_depth_dense.to_sparse()
    #print("sparse tensor indices is shape ", tensor_sparse.indices().shape)
    #print("sanity check: sum of tensor is ", tensor_sparse.sum())
    indices = tensor_sparse.indices()
    values = tensor_sparse.values()
    # Apply the transformations to the coordinates
    # z = values[indices] / depth_scale # = transformed values
    # x = (indices[2] - cx) * z / fx
    # y = (indices[1] - cy) * z / fy
    transformed_values = values / depth_scale # z
    transformed_indices = torch.zeros_like(indices, dtype=torch.float32)
    transformed_indices[1] = (indices[2].float() - cx) * transformed_values / fx # x
    transformed_indices[2] = (indices[1].float() - cy) * transformed_values / fy # y, y is first in opencv!!!
    # Apply the transformations to the coordinates
    #print("transformed indices have shape ", transformed_indices.shape)
    # Create a new sparse tensor with the transformed coordinates and values
    transformed_sparse_tensor = torch.sparse_coo_tensor(transformed_indices, transformed_values, masked_depth_dense.size())
    print("pointcloud batch is of shape ", transformed_sparse_tensor.shape)
    return transformed_sparse_tensor
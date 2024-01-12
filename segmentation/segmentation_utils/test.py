import torch
channel_number = 3
mask_number = 30
u = 1280
v = 720
# Create a tensor of size 3x720x1280 with all elements set to 0.5
pointcloud_tensor = torch.full((channel_number, v, u), 0.5)

# Create a tensor of size 30x720x1280 with sparse filling of 1s
indices = torch.zeros(mask_number, v*u)
values = torch.ones()
# Convert to COO sparse tensor
mask_tensor = torch.sparse_coo_tensor(indices, values, size=(mask_number, v ,u))
# Reshape the pointcloud tensor to match the required shape
pointcloud_tensor = pointcloud_tensor.unsqueeze(0).expand(mask_number, -1, -1, -1)
print(pointcloud_tensor.shape)

# Multiply pointcloud_tensor and mask_tensor element-wise to get the final tensor
result_tensor = pointcloud_tensor * mask_tensor.unsqueeze(1)

print(result_tensor.shape)  # Output: torch.Size([30, 3, 720, 1280])
print(result_tensor.to_dense())  # Output: torch.Size([30, 3, 720, 1280])



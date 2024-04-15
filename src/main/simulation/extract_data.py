
# with h5py.File('navier_stokes_cylinder/velocity.h5', 'r') as f: 
#     group = f['/VisualisationVector'] #dataset p 1D v 3D 999 datasets
#     group_2 = f['/Mesh'] #subgroup
  

#     # V = 
#     # for name, data in group.items():
        
        
#     #     print("\nData from member '{}'".format(name))
#     #     # Check if the member is a dataset
#     #     if isinstance(data, h5py.Dataset):
#     #         # print("Data shape:", data.shape)
#     #         # print("Data type:", data.dtype)
#     #         print("Data:", data[:])  # Read all data from the dataset
        
#     #     else:
#     #         # If it's a subgroup, you can access its members recursively
#     #         print("This member is a subgroup.")

#     # for name in group.items():
#     #     print(name)

#     for j, data in group.items():
#         if int(j) == 0 :
#             plt.figure()
#             plt.scatter(data[:, 0], data[:, 1], color = 'b')
#             plt.savefig(f'plot_{j}.png')
#             plt.show
#         else:
#             pass

#     for j, data in group_2.items():
#         if int(j) == 0 :
#             print(data['mesh/geometry'][:])
                
           
#             plt.figure()
#             plt.scatter(data['mesh/geometry'][:, 0],data['mesh/geometry'][:, 1], color = 'b')
#             plt.savefig(f'mesh_plot_{j}.png')
#             plt.show
#         else:
#             pass


    # V = 
    # for name, data in group.items():
        
        
    #     print("\nData from member '{}'".format(name))
    #     # Check if the member is a dataset
    #     if isinstance(data, h5py.Dataset):
    #         # print("Data shape:", data.shape)
    #         # print("Data type:", data.dtype)
    #         print("Data:", data[:])  # Read all data from the dataset
        
    #     else:
    #         # If it's a subgroup, you can access its members recursively
    #         print("This member is a subgroup.")

    # for name in group.items():
    #     print(name)

    # for j, data in group.items():
    #     if int(j) == 0 :
    #         plt.figure()
    #         plt.plot(data[:, 0], color = 'b')
    #         plt.savefig(f'p_plot_{j}.png')
    #         plt.show
    #     else:
    #         pass

    # for j, data in group_2.items():
    #     if int(j) == 0 :
    #         print(data['mesh/geometry'][:])
                
           
    #         plt.figure()
    #         plt.scatter(data['mesh/geometry'][:, 0],data['mesh/geometry'][:, 1], color = 'b')
    #         plt.savefig(f'mesh_plot_{j}.png')
    #         plt.show
    #     else:
    #         pass


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image


# with h5py.File('navier_stokes_cylinder/velocity.h5', 'r') as f: 
#     group = f['/VisualisationVector'] #dataset p 1D v 3D 999 datasets
#     group_2 = f['/Mesh'] #subgroup
#     dataset_list = []
#     i = 0

#     for j, data in group.items():
#         i += 1
#         while i < 1000:
#             dataset_list.append(data[:, 0:2])

#     dataset_array = np.array(dataset_list)
    
# # Create figure and axis
# fig, ax = plt.subplots()
# scatter = ax.scatter([], [])

# # Update function for animation
# def update(frame):
#     scatter.set_offsets(dataset_array[frame, :, :])
#     return scatter,

# # Create animation
# ani = FuncAnimation(fig, update, frames=dataset_array.shape[0], interval=50, blit=True)



# # Save animation as GIF
# ani.save('animated_scatter.gif', writer='pillow')



# import numpy as np
# from fenics import *


# # Initialize data structure to store solution vectors
# velocity = []

# with h5py.File('navier_stokes_cylinder/velocity.h5', 'r') as f: 
#     group = f['/VisualisationVector'] 
#     for name, data in group.items():
#         velocity.append(data[:,0:2])
#     velocity_array = np.array(velocity)
#     velocity_array = np.transpose(velocity_array, (1, 2, 0))
       

# fig, ax = plt.subplots()
# scatter = ax.scatter([], [])
# ax.set_xlim(0, 2.2)
# ax.set_ylim(-0.2, 0.2)

# # Update function for animation
# def update(frame):
#     scatter.set_offsets(velocity_array[frame, :, :])
#     return scatter,

# # Create animation
# ani = FuncAnimation(fig, update, frames=velocity_array.shape[0], interval=50, blit=True)

# # # Save animation as GIF
# ani.save('animated_scatter.gif', writer='pillow')



# # Reshape solution array to desired format (Nx2xT)
# # Replace 'N', 'T' with actual values
# N = velocity_array.shape[1]  # Number of data points
# T = velocity_array.shape[0]  # Number of time steps
# solution_reshaped = velocity_array.reshape((N, 2, T))

import pickle

with open('dataset.pkl', 'rb') as f:
    loaded_output = pickle.load(f)


times = []
x_coords = []
y_coords = []
velocities = []
pressures = []



for data in loaded_output[0:2]:
    t, x, y, v, p = data  
    times.append(t)
    velocities.append(v)
    pressures.append(p)

    if len(x_coords) == 0:
        x_coords.append(x)
        y_coords.append(y)

print(len(y))


# Create a function to update the plot for each frame of the animation
# def animate(i):
#     plt.clf()
#     plt.scatter(x_coords[i], y_coords[i], c=velocities[i], cmap='coolwarm', vmin=min(velocities), vmax=max(velocities))
#     plt.xlabel('Time')
#     plt.ylabel('Velocity')
#     plt.title('Velocity vs. Time')
#     plt.grid(False)

# # Create the figure and axis objects
# fig = plt.figure()

# ani = FuncAnimation(fig, animate, frames=len(times), interval=200)

# # # Save the animation as a GIF
# ani.save('velocity_vs_time.gif', writer='imagemagick')

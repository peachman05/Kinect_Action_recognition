import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.animation as animation


N_points = 10


def update(num, my_ax):
    # the following corresponds to whatever logic must append in your code
    # to get the new coordinates of your points
    # in this case, we're going to move each point by a quantity (dx,dy,dz)
    dx, dy, dz = np.random.normal(size=(3,N_points), loc=0, scale=1) 
    debug_text.set_text("{:d}".format(num))  # for debugging
    x,y,z = graph._offsets3d
    new_x, new_y, new_z = (x+dx, y+dy, z+dz)
    graph._offsets3d = (new_x, new_y, new_z)
    for t, new_x_i, new_y_i, new_z_i in zip(annots, new_x, new_y, new_z):
        # animating Text in 3D proved to be tricky. Tip of the hat to @ImportanceOfBeingErnest
        # for this answer https://stackoverflow.com/a/51579878/1356000
        x_, y_, _ = proj3d.proj_transform(new_x_i, new_y_i, new_z_i, my_ax.get_proj())
        t.set_position((x_,y_))
    return [graph,debug_text]+annots


# create N_points initial points
x,y,z = np.random.normal(size=(3,N_points), loc=0, scale=10)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
graph = ax.scatter(x, y, z, color='orange')
debug_text = fig.text(0, 1, "TEXT", va='top')  # for debugging
annots = [ax.text2D(0,0,"POINT") for _ in range(N_points)] 

# Creating the Animation object
ani = animation.FuncAnimation(fig, update, fargs=[ax], frames=100, interval=50, blit=True)
plt.show()
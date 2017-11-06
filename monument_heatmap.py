import bpy
import bmesh
import os
import sys
import csv
import numpy as np
from pyproj import Proj
from mathutils import Matrix, Vector
from matplotlib import cm


# Check if script is executed in Blender and get absolute path of current folder
if bpy.context.space_data is not None:
    cwd = os.path.dirname(bpy.context.space_data.text.filepath)
else:
    cwd = os.path.dirname(os.path.abspath(__file__))

sys.path.append(cwd)
os.chdir(cwd)


def bmeshToObject(bm, name='Object'):
    mesh = bpy.data.meshes.new(name+'Mesh')
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.objects.link(obj)
    bpy.context.scene.update()

    return obj

def trackToConstraint(obj, target):
    constraint = obj.constraints.new('TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

def createTarget(origin=(0,0,0)):
    tar = bpy.data.objects.new('Target', None)
    bpy.context.scene.objects.link(tar)
    tar.location = origin

    return tar

def createCamera(origin, target=None, lens=35, clip_start=0.1, clip_end=200, type='PERSP', ortho_scale=6):
    # Create object and camera
    camera = bpy.data.cameras.new("Camera")
    camera.lens = lens
    camera.clip_start = clip_start
    camera.clip_end = clip_end
    camera.type = type # 'PERSP', 'ORTHO', 'PANO'
    if type == 'ORTHO':
        camera.ortho_scale = ortho_scale

    # Link object to scene
    obj = bpy.data.objects.new("CameraObj", camera)
    obj.location = origin
    bpy.context.scene.objects.link(obj)
    bpy.context.scene.camera = obj # Make this the current camera

    if target: trackToConstraint(obj, target)
    return obj

def createLamp(origin, type='POINT', energy=1, color=(1,1,1), target=None):
    # Lamp types: 'POINT', 'SUN', 'SPOT', 'HEMI', 'AREA'
    bpy.ops.object.add(type='LAMP', location=origin)
    obj = bpy.context.object
    obj.data.type = type
    obj.data.energy = energy
    obj.data.color = color

    if target: trackToConstraint(obj, target)
    return obj

def simpleMaterial(diffuse_color):
    mat = bpy.data.materials.new('Material')

    # Diffuse
    mat.diffuse_shader = 'LAMBERT'
    mat.diffuse_intensity = 0.9
    mat.diffuse_color = diffuse_color

    # Specular
    mat.specular_intensity = 0

    return mat


def barplot(data):
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    bm = bmesh.new()

    h, size, scale = 10, 0.1, 5
    for idx in indices:
        x, y, z = data[idx]

        T = Matrix.Translation((scale*(x - 0.5), scale*(y - 0.5), z*(h - 1)*0.5*size))
        S = Matrix.Scale((h - 1)*z + 1, 4, (0, 0, 1))
        bmesh.ops.create_cube(bm, size=size, matrix=T*S)

    obj = bmeshToObject(bm)

    # Add bevel modifier
    bevel = obj.modifiers.new('Bevel', 'BEVEL')
    bevel.width = 0.008

    return obj

def histogram(data, n=50):
    indices = np.arange(len(data))
    #np.random.shuffle(indices)

    bm = bmesh.new()
    h, size, scale = 10, 0.08, 5
    X = np.zeros((n, n))
    for idx in indices:
        x, y, z = data[idx]
        i, j = int(x * (n - 1)), int(y * (n - 1))
        X[i, j] += z

    for i in range(n):
        for j in range(n):
            x, y, z = i / (n - 1), j / (n - 1), X[i, j]

            if z > 0:
                T = Matrix.Translation((
                    scale*(x - 0.5),
                    scale*(y - 0.5),
                    z*(h - 1)*0.5*size))
                S = Matrix.Scale((h - 1)*z + 1, 4, (0, 0, 1))
                bmesh.ops.create_cube(bm, size=size, matrix=T*S)

    obj = bmeshToObject(bm)

    # Add bevel modifier
    bevel = obj.modifiers.new('Bevel', 'BEVEL')
    bevel.width = 0.008

    return obj

def heatmap(data, n=100, m=2):
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    bm = bmesh.new()
    h, size, scale = 15, 0.04, 5
    X = np.ndarray((n, n), dtype=object)
    for idx in indices:
        x, y, z = data[idx]
        i, j = int(x * (n - 1)), int(y * (n - 1))
        if X[i, j] is None:
            X[i, j] = [(x, y, z)]
        else:
            X[i, j].append((x, y, z))

    sigmaSq = 0.0001
    #sigmaSq = 0.0003
    for i0 in range(n):
        for j0 in range(n):
            x0, y0 = i0 / (n - 1), j0 / (n - 1)

            z = 0
            for i in range(max(0, i0 - m), min(i0 + m, n)):
                for j in range(max(0, j0 - m), min(j0 + m, n)):
                    if X[i, j] is not None:
                        for x, y, z0 in X[i, j]:
                            z += z0*np.exp( - ((x0 - x)**2)/(2*sigmaSq) \
                                            - ((y0 - y)**2)/(2*sigmaSq) )

            if z > 0.01:
                T = Matrix.Translation((
                    scale*(x0 - 0.5),
                    scale*(y0 - 0.5),
                    z*(h - 1)*0.5*size))
                S = Matrix.Scale((h - 1)*z + 1, 4, (0, 0, 1))
                bmesh.ops.create_cube(bm, size=size, matrix=T*S)

    obj = bmeshToObject(bm)

    # Add bevel modifier
    bevel = obj.modifiers.new('Bevel', 'BEVEL')
    bevel.width = 0.008

    return obj

def coloredHeatmap(data, n=100, m=2, numColors=10, colormap=cm.hot):
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    # List of bmesh elements for each color group
    bmList = [bmesh.new() for i in range(numColors)]

    h, size, scale = 15, 0.04, 5
    X = np.ndarray((n, n), dtype=object)
    for idx in indices:
        x, y, z = data[idx]
        i, j = int(x * (n - 1)), int(y * (n - 1))
        if X[i, j] is None:
            X[i, j] = [(x, y, z)]
        else:
            X[i, j].append((x, y, z))

    sigmaSq = 0.0001
    #sigmaSq = 0.0003
    grid = np.zeros((n, n))

    for i0 in range(n):
        for j0 in range(n):
            x0, y0 = i0 / (n - 1), j0 / (n - 1)

            # Sum all available neighboring elements
            for i in range(max(0, i0 - m), min(i0 + m, n)):
                for j in range(max(0, j0 - m), min(j0 + m, n)):
                    if X[i, j] is not None:
                        for x, y, z in X[i, j]:
                            grid[i0][j0] += z*np.exp(- ((x0 - x)**2)/
                                (2*sigmaSq) - ((y0 - y)**2)/(2*sigmaSq))

    # Find maximum value
    zMax = np.max(grid)

    # Iterate over grid
    for i in range(n):
        for j in range(n):
            x, y, z = i / (n - 1), j / (n - 1), grid[i][j]
            if z > 0.01:
                t = 1 - np.exp(-(z / zMax)/0.2)

                k = min(int(numColors*t), numColors - 1)
                T = Matrix.Translation((
                    scale*(x - 0.5),
                    scale*(y - 0.5),
                    z*(h - 1)*0.5*size))
                S = Matrix.Scale((h - 1)*z + 1, 4, (0, 0, 1))
                bmesh.ops.create_cube(bmList[k], size=size, matrix=T*S)

    objList = []
    for i, bm in enumerate(bmList):
        # Create object
        obj = bmeshToObject(bm)

        # Create material with colormap
        color = colormap(i / numColors)
        mat = simpleMaterial(color[:3])
        obj.data.materials.append(mat)
        objList.append(obj)

        # Add bevel modifier
        bevel = obj.modifiers.new('Bevel', 'BEVEL')
        bevel.width = 0.008

    return objList


def loadData(path):

    p = Proj(init="epsg:3785")  # Popular Visualisation CRS / Mercator

    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)

        data = np.array(list(reader))[:, 2:].astype(float)

        # Project coordinates
        data[:, 0], data[:, 1] = p(data[:, 1], data[:, 0])

        # Scale coordinates to 0, 1
        minX, maxX = data[:, 0].min(), data[:, 0].max()
        minY, maxY = data[:, 1].min(), data[:, 1].max()
        rangeX, rangeY = maxX - minX, maxY - minY

        if rangeX > rangeY:
            data[:, 0] = (data[:, 0] - minX - 0.5*rangeX) / rangeX + 0.5
            data[:, 1] = (data[:, 1] - minY - 0.5*rangeY) / rangeX + 0.5
        else:
            data[:, 0] = (data[:, 0] - minX - 0.5*rangeX) / rangeY + 0.5
            data[:, 1] = (data[:, 1] - minY - 0.5*rangeY) / rangeY + 0.5

        # Scale counts to 0, 1
        data[:, 2] -= data[:, 2].min()
        data[:, 2] /= data[:, 2].max()

    return data


if __name__ == '__main__':
    # Remove all elements in scene
    bpy.ops.object.select_by_layer()
    bpy.ops.object.delete(use_global=False)

    # Create scene
    target = createTarget()
    camera = createCamera((0.82, -2.35, 2.6), target,
        type='ORTHO', ortho_scale=5.6)
    sun = createLamp((-5, 5, 10), 'SUN', target=target)

    # Set background color
    bpy.context.scene.world.horizon_color = (0.7, 0.7, 0.7)

    # Ambient occlusion
    bpy.context.scene.world.light_settings.use_ambient_occlusion = True
    bpy.context.scene.world.light_settings.samples = 5

    data = loadData(os.path.join('data', 'monuments.txt'))

    #barplot(data)
    #histogram(data)
    #heatmap(data)
    coloredHeatmap(data, colormap=cm.coolwarm)

    # Set render resolution
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.resolution_percentage = 100

    # check if script is not executed inside Blender and render scene
    if bpy.context.space_data is None:
        bpy.context.scene.render.filepath = os.path.join(
            os.getcwd(), 'monuments_coolwarm.png')
        bpy.ops.render.render(write_still=True)

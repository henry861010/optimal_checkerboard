import sys
sys.path.append("/Users/henry/Desktop/code/optimal_checkerboard/src/optimal_checkerboard/algorithms")
from drag import Dragger

# https://pyvista.org/projects/index.html

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import random

random.seed(1)

class Vision:
    def __init__(self):
        ### 3D elements
        self.comps = {}
        self.elements = np.empty((0, 8), dtype=np.int32)
        self.element_comps = np.empty((0), dtype=np.int32)
        self.nodes = np.empty((0, 3), dtype=np.float64)
    
    def set(self, comps, elements, element_comps, nodes):
        self.elements = elements
        self.element_comps = element_comps
        self.nodes = nodes
        self.comps = comps
        
    def show(self, isRandomColor=False):
        ### Build the cell
        n = self.elements.shape[0]
        cells = np.hstack([np.column_stack([np.full((n,1), 8, dtype=self.elements.dtype), self.elements]).ravel()])
        
        ### Cell types
        celltypes = np.full(n, pv.CellType.HEXAHEDRON, dtype=np.uint8)
        
        ### Create grid
        grid = pv.UnstructuredGrid(cells, celltypes, self.nodes)
        
        ### Attach component ids as cell data for coloring
        grid.cell_data['comp'] = self.element_comps.astype(np.int32)

        ### colors
        if 'comp' in grid.point_data and 'comp' not in grid.cell_data:
            grid = grid.point_data_to_cell_data(pass_point_data=False)
        comp = grid.cell_data['comp'].astype(int)
        vals, counts = np.unique(comp, return_counts=True)
        base = plt.get_cmap('viridis', len(self.comps))           # a ListedColormap
        palette = [to_hex(c) for c in base.colors]   # ['#1f77b4', '#ff7f0e', ...]
        if isRandomColor:
            random.shuffle(palette)
        colors  = palette[:len(vals)]          # as many as you need
        
        ### Plot
        plotter = pv.Plotter()
        plotter.add_mesh(
            grid,
            scalars='comp',
            categories=True,
            preference='cell',
            cmap=colors,               # <-- hex strings OK
            show_edges=True,
            smooth_shading=False,
            annotations={int(v): f"{int(v)}" for v in vals},
            show_scalar_bar=False
        )
        
        legend = [[f"comp {v}: {int(c)} elems", colors[i]] for i, (v, c) in enumerate(zip(vals, counts))]
        legend = [[f"Node Num: {len(self.nodes)}", "black"]] + legend
        legend = [[f"Elem Num: {len(self.elements)}", "black"]] + legend
        
        plotter.add_legend(legend, loc='upper left', bcolor='white', border=True) 
        plotter.add_axes()
        plotter.show()


ELEM_DIM = 4
NODE_DIM = 3
class Mesh2D:
    def __init__(self):
        ### process
        self.elements = np.empty((0, ELEM_DIM), dtype=np.int32)
        self.nodes = np.empty((0, NODE_DIM), dtype=np.float64)
        
        ### others
        self.node_map = {}
        self.node_num = 0
        self.element_num = 0
        
    def pre_allocate_nodes(self, size: int = 1):
        '''
        Exponential growth (1.5x) is a standard amortized allocation method (like Python lists or C++ vectors).
        This avoids frequent resizing for small additions, keeping total reallocation count logarithmic in size.
        Keeps allocation tight when near expected final size.
        '''
        required = self.node_num + size
        current_capacity = len(self.nodes)
        if required > current_capacity:
            new_capacity = max(required, int(current_capacity * 1.5))
            extra = new_capacity - current_capacity
            self.nodes = np.vstack([self.nodes, np.empty((extra, NODE_DIM), dtype=np.float64)])

    def pre_allocate_elements(self, size: int = 1):
        '''
        Exponential growth (1.5x) is a standard amortized allocation method (like Python lists or C++ vectors).
        This avoids frequent resizing for small additions, keeping total reallocation count logarithmic in size.
        Keeps allocation tight when near expected final size.
        '''
        required = self.element_num + size
        current_capacity = len(self.elements)
        if required > current_capacity:
            new_capacity = max(required, int(current_capacity * 1.5))
            extra = new_capacity - current_capacity
            self.elements = np.vstack([self.elements, np.empty((extra, ELEM_DIM), dtype=np.int32)])
        
    ### wrap operation
    def mesh_checkerboard(self, element_size, x_list, y_list):
        # --- validate element_size ---
        if not np.isscalar(element_size) or float(element_size) <= 0:
            raise ValueError("element_size must be a positive scalar.")
        h = float(element_size)

        # --- normalize inputs ---
        x_list = np.asarray(x_list, dtype=np.float64).ravel()
        y_list = np.asarray(y_list, dtype=np.float64).ravel()
        if x_list.ndim != 1 or y_list.ndim != 1 or x_list.size < 2 or y_list.size < 2:
            raise ValueError("x_list and y_list must be 1D arrays with length >= 2.")

        # --- monotonic check ---
        if np.any(np.diff(x_list) < 0) or np.any(np.diff(y_list) < 0):
            raise ValueError("x_list and y_list must be non-decreasing.")

        # --- densify helper (handles zero-length spans) ---
        def densify(arr):
            out = []
            for a, b in zip(arr[:-1], arr[1:]):
                length = float(b - a)
                if length == 0.0:
                    # duplicate line: keep only one point when joining segments
                    if not out:
                        out.append(a)
                    continue
                nseg = max(1, int(np.ceil(length / h)))
                seg = np.linspace(a, b, nseg + 1, endpoint=True, dtype=np.float64)
                if out:
                    seg = seg[1:]  # avoid boundary duplicate
                out.extend(seg.tolist())
            return np.asarray(out, dtype=np.float64)

        x = densify(x_list)
        y = densify(y_list)
        if x.size < 2 or y.size < 2:
            raise ValueError("After densify, need at least 2 x-lines and 2 y-lines.")

        Nx, Ny = int(x.size), int(y.size)

        # --- nodes (x varies fastest) ---
        X, Y = np.meshgrid(x, y, indexing="xy")
        Z = np.zeros_like(X)
        nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)  # (Ny*Nx, 2)

        # --- element node ids ---
        ix = np.arange(Nx - 1, dtype=np.int32)
        iy = np.arange(Ny - 1, dtype=np.int32)
        GX, GY = np.meshgrid(ix, iy, indexing="xy")

        n00 = (GY    ) * Nx + (GX    )  # BL
        n10 = (GY    ) * Nx + (GX + 1)  # BR
        n11 = (GY + 1) * Nx + (GX + 1)  # TR
        n01 = (GY + 1) * Nx + (GX    )  # TL

        # CLOCKWISE: BL, TL, TR, BR
        elements = np.stack([n00, n01, n11, n10], axis=-1).reshape(-1, 4).astype(np.int32)
        elements += self.node_num

        new_node_num = len(nodes)
        new_elem_num = len(elements)
        self.pre_allocate_elements(new_elem_num)
        self.pre_allocate_nodes(new_node_num)
        self.elements[self.element_num:self.element_num+new_elem_num+1] = elements
        self.nodes[self.node_num:self.node_num+new_node_num+1] = nodes
        self.element_num += len(elements)
        self.node_num += len(nodes)

      
mesh2D_obj = Mesh2D()
mesh2D_obj.mesh_checkerboard(1,[0,100],[0,100])

dragger_obj = Dragger()
dragger_obj.set_2D(mesh2D_obj)

area1 = {
    "type": "BOX",
    "dim": [0,0,100,100],
    "holes": [{
        "type": "BOX",
        "dim": [0,0,40,40]
    }],
    "metals": [
        {
            "type": "NORMAL",
            "material": "metal1",
            "density": 10,
            "holes":[
                {
                    "type": "BOX",
                    "dim": [40, 40, 70, 70]
                }
            ]
        }, {
            "type": "NORMAL",
            "material": "metal2",
            "density": 50,
            "ranges": [
                {
                    "type": "BOX",
                    "dim": [70, 70, 90, 90]
                }
            ]
        }
    ],
    "material": "comp1"
}
dragger_obj._organize(area1)
dragger_obj._drag(1,0,5)

area2 = {
    "type": "BOX",
    "dim": [0,0,100,100],
    "material": "comp2",
    "metals": [
        {
            "type": "CONTINUE",
            "material": "metal1"
        }
    ]
}
dragger_obj._organize(area2)
dragger_obj._drag(1,5,10)

area3 = {
    "type": "BOX",
    "dim": [50,50,100,100],
    "material": "comp3",
    "metals": [
        {
            "type": "CONVERT",
            "material_o": "metal1",
            "material": "metal3"
        }
    ]
}
dragger_obj._organize(area3)
dragger_obj._drag(5,10,20)


area4 = {
    "type": "POLYGON",
    "dim": [[[20,60], [40,60], [40,80], [60,80], [90,10]]],
    "material": "comp4"
}
dragger_obj._organize(area4)
area4 = {
    "type": "BOX",
    "dim": [0, 0, 100, 100],
    "material": "EMPTY",
    "holes": [{
        "type": "POLYGON",
        "dim": [[[20,60], [40,60], [40,80], [60,80], [90,10]]],
    }]
}
dragger_obj._organize(area4)
dragger_obj._drag(1,20,30)

vision_obj = Vision()

nodes = dragger_obj.nodes[:dragger_obj.node_num]
elements = dragger_obj.elements[:dragger_obj.element_num]
element_comps = dragger_obj.element_comps[:dragger_obj.element_num]
comps = dragger_obj.comps

vision_obj.set(comps, elements, element_comps, nodes)
vision_obj.show()

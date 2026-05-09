import time
import numpy as np
from matplotlib.path import Path

from optimal_checkerboard.algorithms.polygon import normalize_polygon_loops

'''
    OBJECTIVE: 
        1. Used for the 2.5D Auotomation Framework (engine.py)
'''

ELEMENT_LEN = 8
NODE_LEN = 3


def _search_polygon_element(x4, y4, dim, eps=0.0):
    loops = normalize_polygon_loops(dim)
    points = np.stack((x4, y4), axis=-1).reshape(-1, 2)

    inside_hull = np.zeros(len(points), dtype=bool)
    inside_hole = np.zeros(len(points), dtype=bool)
    for loop in loops:
        loop_mask = _points_in_loop_inclusive(points, loop["points"], eps=eps)
        if loop["role"] == "hull":
            inside_hull |= loop_mask
        else:
            inside_hole |= loop_mask

    point_mask = inside_hull & ~inside_hole
    return point_mask.reshape(len(x4), 4).all(axis=1)


def _points_in_loop_inclusive(points, loop, eps=0.0):
    vertices = np.asarray(loop, dtype=float)
    path_mask = Path(vertices).contains_points(points)
    boundary_mask = _points_on_loop_boundary(points, vertices, eps=eps)
    return path_mask | boundary_mask


def _points_on_loop_boundary(points, vertices, eps=0.0, chunk_size=65536):
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    tol = _polygon_boundary_tolerance(points, vertices, eps)
    x1 = vertices[:, 0]
    y1 = vertices[:, 1]
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)
    dx = x2 - x1
    dy = y2 - y1
    cross_tol = tol * np.maximum(np.maximum(np.abs(dx), np.abs(dy)), 1.0)

    result = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        chunk = points[start:end]
        x = chunk[:, 0:1]
        y = chunk[:, 1:2]
        cross = (x - x1) * dy - (y - y1) * dx
        within_x = (x >= np.minimum(x1, x2) - tol) & (
            x <= np.maximum(x1, x2) + tol
        )
        within_y = (y >= np.minimum(y1, y2) - tol) & (
            y <= np.maximum(y1, y2) + tol
        )
        result[start:end] = np.any(
            (np.abs(cross) <= cross_tol) & within_x & within_y,
            axis=1,
        )
    return result


def _polygon_boundary_tolerance(points, vertices, eps):
    if eps:
        return float(eps)

    scale = max(
        float(np.max(np.abs(points))) if len(points) else 0.0,
        float(np.max(np.abs(vertices))) if len(vertices) else 0.0,
        1.0,
    )
    return 1e-12 * scale


def search_face_element(element_coordinates, type, dim, index=None, eps=0.0, returnMask=False):
    """
        Fast predicate on subset indices (index). 
        Returns a boolean mask aligned to index (or to all rows if index is None).
    """
    rows = index if index is not None else np.arange(len(element_coordinates))

    # gather 4 corners
    cols = np.array([0, 2, 4, 6], dtype=int)
    x4 = element_coordinates[np.ix_(rows, cols)]
    cols = np.array([1, 3, 5, 7], dtype=int)
    y4 = element_coordinates[np.ix_(rows, cols)]
    
    ### max/min of each element_coordinates
    min_x = x4.min(axis=1)
    max_x = x4.max(axis=1)
    min_y = y4.min(axis=1)
    max_y = y4.max(axis=1)

    if type == "BOX":
        bl_x, bl_y, tr_x, tr_y = dim
        if eps:
            bl_x -= eps
            bl_y -= eps
            tr_x += eps
            tr_y += eps
        res_mask = (min_x >= bl_x) & (max_x <= tr_x) & (min_y >= bl_y) & (max_y <= tr_y)

    elif type == "CYLINDER":
        cx, cy, r = dim
        rr = r*r + 0.0
        if eps:
            rr = (r + eps) * (r + eps)
        dist = (x4 - cx)**2 + (y4 - cy)**2
        res_mask = np.all(dist <= rr, axis=1)
    
    elif type == "POLYGON":
        res_mask = _search_polygon_element(x4, y4, dim, eps=eps)
    else:
        raise ValueError(f"Unsupported type: {type}")
    
    if returnMask:
        return res_mask
    else:
        return np.flatnonzero(res_mask) 
    
class Dragger:
    def __init__(self):
        ### component
        self.comps = {"EMPTY":0}
        
        ### 3D elements
        self.element_num = 0
        self.elements = np.empty((0, ELEMENT_LEN), dtype=np.int32)
        self.element_ids = np.empty((0), dtype=np.int32)
        self.element_comps = np.empty((0), dtype=np.int32)
        
        ### nodes
        self.node_num = 0
        self.nodes = np.empty((0, NODE_LEN), dtype=np.float64)
        self.node_ids = np.empty((0), dtype=np.float64)
        
        ### process
        self.element_2D = np.zeros((0, 4), dtype=np.int32)
        self.element_2D_volumn = np.empty((0), dtype=np.float64)
        self.element_2D_comp = np.empty((0), dtype=np.int32)
        
        self.node_2D = np.empty((0, 2), dtype=np.float64)
        self.node_2D_to_3D = np.zeros((0), dtype=np.int32)
        
    ### initial
    def set_2D(self, mesh2D:'Mesh2D'):
        if hasattr(mesh2D, "get_byIndex"):
            nodes, elements = mesh2D.get_byIndex()
        else:
            nodes, elements = mesh2D.nodes, mesh2D.elements

        nodes = np.asarray(nodes, dtype=np.float64)
        elements = np.asarray(elements, dtype=np.int32)
        if nodes.ndim != 2 or nodes.shape[1] < 2:
            raise ValueError("mesh2D.nodes must have shape (n, 2+)") 
        if elements.ndim != 2 or elements.shape[1] != 4:
            raise ValueError("mesh2D.elements must have shape (m, 4)")
        
        self.element_2D = elements
        self.element_2D_comp = np.zeros(len(elements), dtype=np.int32)
        self.element_2D_volumn = np.empty(len(elements), dtype=np.float64)
        self.node_2D = nodes[:,:2]
        self.node_2D_to_3D = np.zeros(len(nodes), dtype=np.int32) - 1
        
        self._cal_volumns()
        
    ### foundmental
    def _pre_allocate_nodes(self, size: int = 1):
        required = self.node_num + size
        current_capacity = len(self.nodes)
        if required > current_capacity:
            new_capacity = max(required, int(current_capacity * 1.5))
            extra = new_capacity - current_capacity
            
            self.nodes = np.vstack([self.nodes, np.empty((extra, 3), dtype=np.float64)])
            self.node_ids = np.concatenate([self.node_ids, np.empty(extra, dtype=np.int32)])

    def _pre_allocate_elements(self, size: int = 1):
        required = self.element_num + size
        current_capacity = len(self.elements)
        if required > current_capacity:
            new_capacity = max(required, int(current_capacity * 1.5))
            extra = new_capacity - current_capacity
            
            self.elements = np.vstack([self.elements, np.empty((extra, 8), dtype=np.int32)])
            self.element_ids = np.concatenate([self.element_ids, np.empty(extra, dtype=np.int32)])
            self.element_comps = np.concatenate([self.element_comps, np.empty(extra, dtype=np.int32)])
        
    ### core
    def _normalize_element_indices(self, element_indices=None):
        if element_indices is None:
            return np.arange(len(self.element_2D), dtype=np.int32)

        element_indices = np.asarray(element_indices, dtype=np.int32)
        if element_indices.ndim == 0:
            element_indices = element_indices.reshape(1)
        return element_indices

    def _element_coordinates(self, element_indices=None):
        element_indices = self._normalize_element_indices(element_indices)
        if len(element_indices) == 0:
            return np.empty((0, 8), dtype=self.node_2D.dtype)

        corner_xy = self.node_2D[self.element_2D[element_indices]]
        return corner_xy.reshape(len(element_indices), 8)

    def _search_faces(self, element_indices=None, ranges=None, holes=None, returnMask=False):
        """
        Ranges-first progressive search using only index arrays (no big copies).
        Returns local indices over 'element_indices'. If element_indices is
        None, the returned indices are global 2D element indices.
        """
        element_indices = self._normalize_element_indices(element_indices)
        n = len(element_indices)
        if n == 0:
            if returnMask:
                return np.zeros(0, dtype=bool)
            else:
                return np.zeros(0, dtype=np.int32)

        element_coordinates = self._element_coordinates(element_indices)
        included_mask = np.zeros(n, dtype=bool)

        ### Include
        if ranges:
            canidate_indices = np.arange(n, dtype=np.int32)
            for r in ranges:
                if len(canidate_indices) == 0:
                    break
                submask = search_face_element(element_coordinates, r["type"], r["dim"], index=canidate_indices, returnMask=True)
                if np.any(submask):
                    hit_indices = canidate_indices[submask]
                    included_mask[hit_indices] = True
                    canidate_indices = canidate_indices[~submask]  # short-circuit: drop already-included_mask
        else:
            included_mask[:] = True

        ### Exclude
        if holes:
            live_indices = np.nonzero(included_mask)[0]
            for h in holes:
                if len(live_indices) == 0:
                    break
                submask = search_face_element(element_coordinates, h["type"], h["dim"], index=live_indices, returnMask=True)
                if np.any(submask):
                    lose_indices = live_indices[submask]
                    included_mask[lose_indices] = False
                    live_indices = live_indices[~submask]
        if returnMask:
            return included_mask
        else:
            return np.flatnonzero(included_mask) 
        
    def _assign_metal(self, volumes, density, total_volume, randomSeed=1):
        """
        Randomly pick elements until reaching density% of total_volume.
        Operates by shuffling indices only and using cumsum to avoid Python loops.
        Returns the chosen *row indices within this subset* (not global IDs).
        """
        if density == 0:
            return np.empty((0), dtype=np.int32)
            
        volumes = np.asarray(volumes)
        target_indices = np.arange(len(volumes), dtype=np.int32)
        
        # target volume
        target = (density / 100.0) * total_volume

        # random order of candidates (indices only, not rows)
        rng = np.random.default_rng(randomSeed)
        random_indices = target_indices[rng.permutation(len(volumes))]

        # cumulative sum until target
        csum = np.cumsum(volumes[random_indices])
        k = np.searchsorted(csum, target, side="right")  # number to take (may be 0)
        if k > 0:
            chosen_indices  = target_indices[random_indices[:k+1]]
            return chosen_indices
        else:
            # no assignment if density threshold is 0 or vols too small
            return np.empty((0), dtype=np.int32)

    def _organize(self, areas, layer=1):
        if isinstance(areas, dict):
            areas = [areas]
            
        for area in areas:
            ### Select the area once (mask -> indices)
            ranges = [{"type": area["type"], "dim": area["dim"]}]
            holes  = area.get("holes")
            area_indices  = self._search_faces(None, ranges, holes)
            if len(area_indices) == 0:
                continue

            ### Working pool: local indices into area_idx
            remaining_indices = np.arange(len(area_indices), dtype=np.int32)

            ### Volumes for NORMAL metals (within area)
            for metal in area.get("metals", []):
                if metal["type"] == "NORMAL":
                    ranges = metal.get("ranges")
                    holes = metal.get("holes")
                    metal_indices = self._search_faces(area_indices, ranges, holes)
                    vol = self.element_2D_volumn[area_indices[metal_indices]].sum()
                    metal["volumn"] = float(vol)

            ### metal assignment CONTINUE
            for metal in area.get("metals", []):
                if metal["type"] == "CONTINUE":
                    ### find potential assignment area  
                    ranges = metal.get("ranges")
                    holes = metal.get("holes")
                    region_local = self._search_faces(area_indices[remaining_indices], ranges, holes)
                    remaining_target_indices = remaining_indices[region_local]
                
                    ### remove the assignment
                    if len(remaining_target_indices) and metal["material"] in self.comps:
                        comp_id = self.comps[metal["material"]]
                        assigned_mask = (self.element_2D_comp[area_indices[remaining_target_indices]] == comp_id)
                        if np.any(assigned_mask):
                            remaining_assigned_indices = remaining_target_indices[assigned_mask]
                            remaining_indices = np.setdiff1d(remaining_indices, remaining_assigned_indices, assume_unique=False)
            
            ### metal assignment CONVERT
            for metal in area.get("metals", []):
                if metal["type"] == "CONVERT":   
                    ### find potential assignment area  
                    ranges = metal.get("ranges")
                    holes = metal.get("holes")
                    region_local = self._search_faces(area_indices[remaining_indices], ranges, holes)
                    remaining_target_indices = remaining_indices[region_local]  
                                    
                    ### convert the assignment metal & remove the assignment
                    if len(remaining_target_indices) and metal["material_o"] in self.comps:
                        material_old = metal["material_o"]
                        material_new = metal["material"]
                        if material_new not in self.comps: 
                            self.comps[material_new] = len(self.comps)
                        comp_id_old = self.comps[material_old] 
                        comp_id_new = self.comps[material_new]
                        
                        assigned_mask = (self.element_2D_comp[area_indices[remaining_target_indices]] == comp_id_old)
                        if np.any(assigned_mask):
                            remaining_assigned_indices = remaining_target_indices[assigned_mask]
                            self.element_2D_comp[area_indices[remaining_assigned_indices]] = comp_id_new
                            remaining_indices = np.setdiff1d(remaining_indices, remaining_assigned_indices, assume_unique=False)
            
            ### metal assignment Normal
            for metal in area.get("metals", []):
                if metal["type"] == "NORMAL": 
                    ### find potential assignment area  
                    ranges = metal.get("ranges")
                    holes = metal.get("holes")
                    density = metal.get("density")
                    volumn = metal.get("volumn")
                    
                    region_local = self._search_faces(area_indices[remaining_indices], ranges, holes)
                    remaining_target_indices = remaining_indices[region_local]  
                                                
                    ### assign metal
                    if len(remaining_target_indices) and density > 0:
                        ### find the assignment area
                        target_volumes = self.element_2D_volumn[area_indices[remaining_target_indices]]
                        remaining_assigned_indices = self._assign_metal(target_volumes, density, volumn, randomSeed=layer)
                        assigned_indices  = remaining_target_indices[remaining_assigned_indices]
                        
                        ### assigne metal
                        material = metal["material"]
                        if material not in self.comps:
                            self.comps[material] = len(self.comps)
                        comp_id = self.comps[material]

                        ### assign the metal
                        self.element_2D_comp[area_indices[assigned_indices]] = comp_id

                        ### remove assigned element
                        temp_mask = ~np.isin(remaining_indices, assigned_indices) 
                        remaining_indices = remaining_indices[temp_mask]

            ### assign the material
            if len(remaining_indices):
                material = area["material"]
                if material not in self.comps:
                    self.comps[material] = len(self.comps)
                comp_id = self.comps[material]
                self.element_2D_comp[area_indices[remaining_indices]] = comp_id
        
    def _organize_empty(self):
        self.element_2D_comp[:] = 0
        self.node_2D_to_3D[:] = -1
        
    def _drag(self, element_size: float, begin: float, end: float):
        ### calculate drag_num & element_size
        distance  = round(float(end) - float(begin), 5)
        if distance <= 0:
            return 0
        drag_num = int(max(1, np.floor(distance / element_size)))
        element_size = distance / drag_num

        ### target element index
        elem2D_idx = np.flatnonzero(self.element_2D_comp != 0)
        if elem2D_idx.size == 0:
            self.node_2D_to_3D[:] = -1
            return 0

        # unique 2D node ids used by those elements
        elem2D_nodes = self.element_2D[elem2D_idx]
        node2D_idx, inv = np.unique(elem2D_nodes, return_inverse=True)
        elem2D_nodes_local = inv.reshape(elem2D_nodes.shape)   

        ### add the unexisted node
        unexisted_node2D_idx = node2D_idx[self.node_2D_to_3D[node2D_idx] == -1]
        if unexisted_node2D_idx.size:
            self._pre_allocate_nodes(unexisted_node2D_idx.size)
            
            dst = self.nodes[self.node_num : self.node_num + unexisted_node2D_idx.size]
            np.take(self.node_2D, unexisted_node2D_idx, axis=0, out=dst[:, :2]) # xy from 2D nodes -> write directly without temp:
            dst[:, 2] = begin
            
            self.node_2D_to_3D[unexisted_node2D_idx] = np.arange(self.node_num, self.node_num + unexisted_node2D_idx.size, dtype=np.int32)
            self.node_num += unexisted_node2D_idx.size

        ### begin to drag
        base_map = self.node_2D_to_3D[node2D_idx]
        N = node2D_idx.size
        E = elem2D_idx.size

        ### [NODE] allocate all 3D nodes
        self._pre_allocate_nodes(N * drag_num)
        node_start = self.node_num
        new_node_ids = (node_start + np.arange(N * drag_num, dtype=np.int32)).reshape(drag_num, N)

        # fill XY for all layers (broadcast XY), and Z per layer
        xy = self.node_2D[node2D_idx]
        dst_nodes = self.nodes[node_start : node_start + N * drag_num]
        dst_nodes[:, :2] = np.broadcast_to(xy, (drag_num, N, 2)).reshape(N * drag_num, 2)

        z_vals = begin + element_size * (np.arange(1, drag_num + 1, dtype=dst_nodes.dtype))
        dst_nodes[:, 2] = np.repeat(z_vals, N)

        self.node_num += N * drag_num

        ### [ELEMENT] allocate all 3D elements
        self._pre_allocate_elements(E * drag_num)
        elem_start = self.element_num

        # layer->3D index table (drag_num+1, N): row0 = base, rows1..drag_num = new layers
        layer_nodes = np.empty((drag_num + 1, N), dtype=np.int32)
        layer_nodes[0]  = base_map
        layer_nodes[1:] = new_node_ids

        # For each layer k, bottom = layer_nodes[k-1][cols], top = layer_nodes[k][cols]
        # cols per element are the (E,4) indices into node2D_idx
        bottom = layer_nodes[:-1][:, elem2D_nodes_local]
        top    = layer_nodes[1:][:,  elem2D_nodes_local]
        elems  = np.concatenate([bottom, top], axis=2).reshape(drag_num * E, 8)

        ### assign nodes to each element
        self.elements[elem_start : elem_start + drag_num * E] = elems.astype(np.int32, copy=False)
        
        ### assign ids to each element
        last_id = int(np.max(self.element_ids[:self.element_num])) if self.element_num else 0
        self.element_ids[elem_start : elem_start + drag_num * E] = 1 + last_id + np.arange(drag_num * E)

        ### assign comps to each element
        layer_comps = self.element_2D_comp[elem2D_idx]
        dest = self.element_comps[elem_start : elem_start + drag_num * E].reshape(drag_num, E)
        dest[:] = layer_comps
        
        self.element_num += drag_num * E

        # update map so future ops start from the top layer
        self.node_2D_to_3D[:] = -1
        self.node_2D_to_3D[node2D_idx] = layer_nodes[-1]

    def build(self, object_list):
        for index, obj in enumerate(object_list):
            self._organize_empty()
            for index, layer in enumerate(obj[:-1]):
                self._organize(layer["areas"], index)
                self._drag(layer["element_size"], obj[index]["z"], obj[index+1]["z"])
                
    def _cal_volumns(self):
        corner_xy = self.node_2D[self.element_2D]

        # Extract coordinates as (N, 4) for each x and y
        x1, y1 = corner_xy[:, 0, 0], corner_xy[:, 0, 1]
        x2, y2 = corner_xy[:, 1, 0], corner_xy[:, 1, 1]
        x3, y3 = corner_xy[:, 2, 0], corner_xy[:, 2, 1]
        x4, y4 = corner_xy[:, 3, 0], corner_xy[:, 3, 1]

        # Shoelace formula for quadrilateral
        voulmn = 0.5 * np.abs(
            x1*y2 + x2*y3 + x3*y4 + x4*y1 -
            (y1*x2 + y2*x3 + y3*x4 + y4*x1)
        )

        self.element_2D_volumn = voulmn.astype(np.float64, copy=False)


Engin25D = Dragger

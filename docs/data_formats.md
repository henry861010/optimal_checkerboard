# 輸入與輸出資料格式

這份文件定義 `optimal_checkerboard` 對外使用時的資料契約。演算法原理請參考 [演算法與資料流](optimal_checkerboard_algorithm.md)。

## 1. 座標與共同規則

- XY 使用 Cartesian coordinates。
- Z 表示垂直堆疊方向。
- 數值座標可以是 `int` 或 `float`，內部通常會轉成 `float`。
- pattern feature edge 必須是水平或垂直。
- 所有 Z interval 必須滿足 `bottom_z <= top_z` 或 `begin <= end`。
- `POLYGON` 可以省略重複的結尾點；程式會自動連接最後一點與第一點。
- `POLYGON` 使用 winding 表示 loop role：
  - 順時針：hull。
  - 逆時針：hole。

## 2. 公開幾何輸入：`Obj`

正常 client 應使用：

```python
mesher.set_pattern_obj(obj, element_size, ratio=0.1)
```

而不是直接建立 raw face dictionaries。

### 2.1 `Obj`

```python
Obj(type, dim, z=0)
```

| 欄位 | 型別 | 意義 |
| --- | --- | --- |
| `type` | `str` | root face type，例如 `BOX`、`POLYGON`、`CYLINDER` |
| `dim` | `list` | 相對於父物件 origin 的 XY 幾何 |
| `z` | number | 相對於父物件的起始 Z |
| `layers` | `list[Layer]` | 材料堆疊 |
| `metals` | `list[Metal]` | 金屬範圍 |
| `meshs` | `list[Mesh]` | 額外 mesh constraints |
| `child_objs` | `list[Obj]` | 子物件 |
| `thk` | number | 所有 layers 厚度總和 |

加入 layer：

```python
obj.add_layer(thk=10, material="SUBSTRATE")
```

加入 child：

```python
obj.add_child(child_obj)
```

`set_pattern_obj()` 會由 root 開始計算 absolute coordinates。BOX 與 CYLINDER child 的 XY origin 以父物件 face 的前兩個 absolute coordinates 為基準；POLYGON parent 則使用第一個 loop 的第一個點作為 child origin。

### 2.2 `Face`

```python
Face(type, dim)
```

#### BOX

```python
Face("BOX", [x1, y1, x2, y2])
```

請使用排序後的矩形 bounds：

```python
Face("BOX", [0, 0, 10, 5])
```

也就是：

```text
x1 <= x2
y1 <= y2
```

root mesh domain 會正規化 bounds，但 area selection 與其他 client data 不一定會自動交換座標，因此所有 BOX 輸入都建議遵守 `[xmin, ymin, xmax, ymax]`。

#### LINE

shared-rail pipeline 的 LINE 格式為：

```python
[x1, y1, x2, y2]
```

LINE 必須水平或垂直：

```python
[4, 0, 4, 10]   # vertical
[0, 3, 10, 3]   # horizontal
```

公開 `Obj` 路徑通常透過 `Mesh.line` 提供 LINE，而不是直接建立 `Face("LINE", ...)`。

#### POLYGON

```python
Face(
    "POLYGON",
    [
        [[x0, y0], [x1, y1], ...],  # loop 1
        [[x0, y0], [x1, y1], ...],  # loop 2
    ],
)
```

範例：

```python
Face(
    "POLYGON",
    [
        # Clockwise hull
        [[0, 0], [0, 10], [10, 10], [10, 0]],

        # Counter-clockwise hole
        [[3, 3], [7, 3], [7, 7], [3, 7]],
    ],
)
```

用於 shared-rail pattern 時，每一條 polygon edge 都必須水平或垂直。

#### CYLINDER

```python
Face("CYLINDER", [center_x, center_y, radius])
```

此 repository 保留 CYLINDER domain dispatcher 與 generator 入口，供外部 cylinder mesher 整合。CYLINDER 不會被轉成 orthogonal shared-rail pattern lines。

### 2.3 `Layer`

通常不直接建立，而是呼叫：

```python
obj.add_layer(thk=5, material="DIE")
```

等價資料：

```python
Layer(
    material="DIE",
    thk=5,
)
```

每加入一層，`Obj.thk` 就增加該層厚度。`Obj` pattern face 的 active Z interval 是：

```text
[obj.z_abs, obj.z_abs + obj.thk]
```

pattern preprocessing 使用 `Obj.layers` 計算厚度，但不會自動將 layer materials 轉成 `build()` 的 area/material stack。

### 2.4 `Metal`

```python
Metal(
    type,
    begin,
    end,
    material,
    material_o=None,
    ranges=None,
    holes=None,
)
```

範例：

```python
metal = Metal(
    "NORMAL",
    begin=2,
    end=8,
    material="M1",
    ranges=[Face("BOX", [2, 2, 9, 7])],
    holes=[Face("BOX", [4, 4, 5, 5])],
)
obj.metals.append(metal)
```

`begin` 和 `end` 是相對於所屬 `Obj` 的 Z。`ranges` 與 `holes` 的邊界會進入 pattern preprocessing；沒有 ranges 或 holes 的 metal 不會額外增加 pattern lines，因為它沿用 parent footprint。

### 2.5 `Mesh`

```python
Mesh(
    begin,
    end,
    element_size=None,
    line=None,
    face=None,
)
```

mesh line：

```python
Mesh(
    begin=0,
    end=10,
    line=[[5, 0], [5, 10]],
)
```

mesh face：

```python
Mesh(
    begin=2,
    end=8,
    face=Face("BOX", [2, 2, 4, 4]),
)
```

`line` 與 `face` 的邊界會在 `[begin_abs, end_abs]` 期間成為 pattern features。

## 3. Private/debug raw face format

測試或演算法除錯可以直接使用：

```python
mesher._set_pattern(
    faces,
    element_size,
    ratio=0.1,
    mesh_domain=None,
)
```

這不是建議的一般 client 入口，但它的資料格式穩定且適合單元測試。

共同格式：

```python
{
    "type": "BOX" | "LINE" | "POLYGON",
    "dim": ...,
    "bottom_z": z0,
    "top_z": z1,
}
```

### BOX raw face

```python
{
    "type": "BOX",
    "dim": [x1, y1, x2, y2],
    "bottom_z": z0,
    "top_z": z1,
}
```

### LINE raw face

```python
{
    "type": "LINE",
    "dim": [x1, y1, x2, y2],
    "bottom_z": z0,
    "top_z": z1,
}
```

### POLYGON raw face

```python
{
    "type": "POLYGON",
    "dim": [
        [[x0, y0], [x1, y1], ...],
        ...
    ],
    "bottom_z": z0,
    "top_z": z1,
}
```

若 raw caller 仍要呼叫 `mesh_checkerboard()`，必須提供 root mesh domain：

```python
mesh_domain = {
    "type": "BOX",
    "dim": [xmin, ymin, xmax, ymax],
}
```

完整範例：

```python
faces = [
    {
        "type": "BOX",
        "dim": [0, 0, 10, 10],
        "bottom_z": 0,
        "top_z": 10,
    },
    {
        "type": "LINE",
        "dim": [4, 0, 4, 10],
        "bottom_z": 2,
        "top_z": 8,
    },
]

mesher._set_pattern(
    faces,
    element_size=2,
    ratio=0.1,
    mesh_domain={"type": "BOX", "dim": [0, 0, 10, 10]},
)
```

## 4. Pattern preprocessing 參數

### `element_size`

`set_pattern_obj()` 與 `_set_pattern()` 的 `element_size` 表示 XY checkerboard mesh 的偏好最大元素尺寸。

在兩條必要 rail 之間，元素數量約為：

```python
ceil(interval_length / element_size)
```

因此實際元素尺寸可能小於 `element_size`，以便剛好落在下一條必要 rail 上。

### `ratio`

rail merge tolerance：

```python
merge_tol = element_size * ratio
```

例如：

```text
element_size = 5.0
ratio        = 0.2
merge_tol    = 1.0
```

只有在幾何與拓撲規則都允許時，座標範圍不超過 `merge_tol` 的 features 才可能共用 rail。

## 5. 自訂 2D mesh 輸入

```python
mesher.mesh_assignment(mesh2d)
```

`mesh2d` 可以是任何具有以下 attributes 的物件：

```python
mesh2d.nodes
mesh2d.elements
```

必要格式：

```text
nodes     shape (n, 2+)  floating point coordinates
elements  shape (m, 4)   integer node ids
```

驗證規則：

- `elements` 的 node id 必須位於 `[0, len(nodes))`。
- 每一條 `mesher.x_list` rail 必須在 nodes 中有對應的 X 座標。
- 每一條 `mesher.y_list` rail 必須在 nodes 中有對應的 Y 座標。
- snapping 可以改變 coordinates，但不應改變 topology 或 node ids。

外部 mesher 可先取得 shared-rail default faces：

```python
snap_faces = mesher.get_snap_faces()
```

這個方法回傳 deep copies，不會修改原始 `mesher.faces`。

## 6. `build()` 的 3D layer stack

```python
dragger = mesher.build(obj_list, preserve_mesh2d=False)
```

`Obj` hierarchy 與 `obj_list` 是兩份不同用途的輸入：

| 輸入 | 責任 |
| --- | --- |
| `Obj` hierarchy | pattern geometry、absolute placement、active Z interval、shared rails |
| `obj_list` | 各 Z interval 的 area/material 與 Z extrusion size |

目前沒有從 `Obj.layers`／`Obj.metals` 自動產生 `obj_list` 的轉換流程。

### 6.1 最外層格式

```python
obj_list = [
    object_stack_1,
    object_stack_2,
    ...
]
```

每個 object stack 是依 Z 遞增排列的 layer event list：

```python
object_stack = [
    layer_0,
    layer_1,
    ...,
    final_z_sentinel,
]
```

至少要有兩個 Z entries。最後一筆主要提供前一層的 `z_end`，不會被 organize 或 drag。

`build()` 只會在每個非 sentinel entry 的 `z` 呼叫 `apply_snap_rules_at_z()`。因此若某個 pattern feature 在 `z=5` 開始，object stack 也必須包含 `{"z": 5, ...}`，否則該 snap event 不會在正確高度執行。一般而言，layer stack 應包含所有幾何、材料與 snap/restore 會改變的 Z events。

### 6.2 一般 layer

```python
{
    "z": z_begin,
    "element_size": z_element_size,
    "areas": [area, ...],
}
```

| 欄位 | 必要性 | 意義 |
| --- | --- | --- |
| `z` | 必要 | 此 layer 的 bottom Z |
| `element_size` | 最後 sentinel 以外必要 | Z 方向偏好 extrusion size |
| `areas` | 最後 sentinel 以外必要 | 此 Z interval 的材料區域 |

下一筆 entry 的 `z` 是目前 layer 的 `z_end`。

### 6.3 Final sentinel

```python
{
    "z": final_z,
}
```

即使 sentinel 含有其他欄位，`build()` 也不會處理其 `areas`。

### 6.4 Area

```python
{
    "type": "BOX" | "POLYGON" | "CYLINDER",
    "dim": ...,
    "material": "MATERIAL_NAME",
    "holes": [...],    # optional
    "metals": [...],   # optional
}
```

BOX 範例：

```python
{
    "type": "BOX",
    "dim": [0, 0, 10, 10],
    "material": "CORE",
}
```

Area selection 以完整 2D quadrilateral element 為單位；元素四個角都位於 area 內時，該元素才會被選取。

`holes` 使用與 area 相同的 face dictionary：

```python
"holes": [
    {
        "type": "BOX",
        "dim": [4, 4, 6, 6],
    }
]
```

### 6.5 Area metals

這裡的 metal dictionaries 屬於 3D material assignment schema，與 `data_structure.Metal` 是不同層次的資料。

#### NORMAL

依 density 隨機選取 candidate elements：

```python
{
    "type": "NORMAL",
    "material": "M1",
    "density": 50,
    "ranges": [...],  # optional
    "holes": [...],   # optional
}
```

`density` 是目標 area volume 的百分比。選取順序使用固定 seed，因此相同 layer/input 應可重現。

#### CONTINUE

保留上一個 Z interval 已經指定為某材料的 elements，不讓後續 area material 覆寫：

```python
{
    "type": "CONTINUE",
    "material": "M1",
    "ranges": [...],  # optional
    "holes": [...],   # optional
}
```

#### CONVERT

將指定範圍內的既有材料轉換為新材料：

```python
{
    "type": "CONVERT",
    "material_o": "M1",
    "material": "M2",
    "ranges": [...],  # optional
    "holes": [...],   # optional
}
```

### 6.6 `preserve_mesh2d`

預設：

```python
preserve_mesh2d=False
```

`Dragger.node_2D` 與 `mesher.mesh2d.nodes` 共用記憶體，snap 會直接修改原本 2D nodes。

若使用：

```python
preserve_mesh2d=True
```

`build()` 會複製一份 nodes 給 `Dragger`，保留原始 `mesher.mesh2d.nodes`。

## 7. 輸出資料格式

### 7.1 `set_pattern_obj()` / `_set_pattern()`

回傳：

```python
(
    group_lines_v,
    group_lines_h,
    x_list,
    y_list,
)
```

| 輸出 | 意義 |
| --- | --- |
| `group_lines_v` | X rails 所代表的 vertical pattern lines，依 Z 分組 |
| `group_lines_h` | Y rails 所代表的 horizontal pattern lines，依 Z 分組 |
| `x_list` | shared X rail coordinates |
| `y_list` | shared Y rail coordinates |

canonical pattern line：

```python
[
    [x1, y1],
    [x2, y2],
    [z_bottom, z_top],
]
```

`x_list`／`y_list` 表示由 pattern features 建立的 shared rails。內建 BOX generator 會再將 root bounds 合併進 `mesher.mesh_x_list`／`mesher.mesh_y_list`，確保外部邊界存在於最終 mesh。

### 7.2 Public rail

`mesher.rails`：

```python
{
    "x": [x_rail, ...],
    "y": [y_rail, ...],
}
```

單一 rail：

```python
{
    "axis": "x",
    "rail_id": 0,
    "coord": 1.25,
    "min_coord": 1.0,
    "max_coord": 1.5,
    "member_count": 2,
    "lines_by_z": {
        0.0: [
            [[1.0, 0.0], [1.0, 5.0], [0.0, 10.0]],
        ],
    },
}
```

### 7.3 Snap/restore rules

取得全部 rules：

```python
snap_rules_by_z = mesher.get_snap_rules()
restore_rules_by_z = mesher.get_restore_rules()
```

取得特定 Z：

```python
rules = mesher.get_snap_rules(z=10.0)
```

單一 rule：

```python
{
    "kind": "snap",
    "axis": "x",
    "rail_id": 0,
    "rail_coord": 1.25,
    "target_coord": 1.0,
    "z": 0.0,
    "z_bottom": 0.0,
    "z_top": 10.0,
    "span_min": 0.0,
    "span_max": 5.0,
    "feature_id": 0,
}
```

restore rule 使用相同欄位，但：

```python
rule["kind"] == "restore"
rule["z"] == rule["z_top"]
rule["target_coord"] == rule["rail_coord"]
```

### 7.4 `apply_snap_rules_at_z()`

```python
touched_count = mesher.apply_snap_rules_at_z(z)
```

或：

```python
touched_count, node_ids = mesher.apply_snap_rules_at_z(
    z,
    return_touched_node_ids=True,
)
```

這個方法會直接修改 nodes array。

### 7.5 `Mesh2D`

```python
mesh2d = mesher.mesh_checkerboard()
```

```text
mesh2d.nodes     shape (n_nodes, 3), float64
mesh2d.elements  shape (n_elements, 4), int32
```

node：

```python
[x, y, z]
```

內建 BOX mesh 的 base Z 為 `0`。

quadrilateral connectivity：

```python
[bottom_left, bottom_right, top_right, top_left]
```

### 7.6 `Dragger`

```python
dragger = mesher.build(obj_list)
```

`Dragger` 使用預先配置的 arrays，因此只應讀取有效範圍：

```python
nodes = dragger.nodes[:dragger.node_num]
elements = dragger.elements[:dragger.element_num]
element_ids = dragger.element_ids[:dragger.element_num]
element_comps = dragger.element_comps[:dragger.element_num]
components = dragger.comps
```

shape：

```text
nodes          (n_nodes, 3)
elements       (n_elements, 8)
element_ids    (n_elements,)
element_comps  (n_elements,)
components     dict[str, int]
```

hexahedral connectivity：

```python
[
    bottom_face_node_0,
    bottom_face_node_1,
    bottom_face_node_2,
    bottom_face_node_3,
    top_face_node_0,
    top_face_node_1,
    top_face_node_2,
    top_face_node_3,
]
```

`components` 預設包含：

```python
{"EMPTY": 0}
```

其他材料會在第一次使用時依序取得 component id。

## 8. 完整最小範例

```python
from optimal_checkerboard import OptimalMesh25D
from optimal_checkerboard.data_structure.geometry import Obj


root = Obj("BOX", [0, 0, 10, 10], z=0)
root.add_layer(thk=5, material="CORE")

mesher = OptimalMesh25D()
mesher.set_pattern_obj(
    root,
    element_size=2,
    ratio=0.1,
)

mesh2d = mesher.mesh_checkerboard()

obj_list = [
    [
        {
            "z": 0,
            "element_size": 1,
            "areas": [
                {
                    "type": "BOX",
                    "dim": [0, 0, 10, 10],
                    "material": "CORE",
                }
            ],
        },
        {"z": 5},
    ]
]

dragger = mesher.build(obj_list)

nodes_2d = mesh2d.nodes
quads_2d = mesh2d.elements

nodes_3d = dragger.nodes[:dragger.node_num]
hexes_3d = dragger.elements[:dragger.element_num]
materials_3d = dragger.element_comps[:dragger.element_num]
```

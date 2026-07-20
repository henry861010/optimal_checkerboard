# optimal_checkerboard

`optimal_checkerboard` 是一套用於 2.5D 幾何的 checkerboard meshing 工具。它特別適合先進封裝或其他「XY 圖形沿 Z 方向分層堆疊」的模型。

核心做法不是為每一層建立完全獨立的 2D 網格，而是：

1. 從所有幾何邊界抽取水平與垂直 feature lines。
2. 將可以安全共用的鄰近 feature lines 合併成 shared rails。
3. 以 shared rails 建立一份可重複使用的 2D quadrilateral mesh。
4. 到達特定 Z layer 時，依 snap rules 將 rail 節點移回真實幾何位置。
5. 將調整後的 2D mesh 沿 Z 方向拉伸成 3D hexahedral mesh。

這種方法的目的是減少 XY grid lines 和 3D 元素數量，同時保留各層真實的幾何邊界。

## 文件導覽

- [輸入與輸出資料格式](docs/data_formats.md)：公開 `Obj` 模型、raw face、`build()` layer stack、NumPy mesh arrays。
- [演算法與資料流](docs/optimal_checkerboard_algorithm.md)：feature extraction、shared rail grouping、snap/restore 與 3D drag。
- [完整 client 範例](script/client_example.py)：建立 `Obj`、產生 2D mesh、逐層套用 snap rules。
- [3D build 範例](script/example_geometry.py)：指定 layer areas/materials 並產生 3D mesh。

## 支援範圍

| 功能 | 狀態 |
| --- | --- |
| `BOX` pattern face | 支援 |
| 水平／垂直 `LINE` pattern face | 支援 |
| orthogonal `POLYGON` pattern face | 支援 |
| `BOX` root footprint 的 2D mesh | 內建支援 |
| 自訂 2D mesh | 可透過 `mesh_assignment()` 指派 |
| `CYLINDER` root dispatcher | 保留為外部 cylinder mesher 的整合接口 |
| `CYLINDER` pattern edge | 不進入 shared-rail 演算法 |
| `POLYGON` root footprint 的內建 2D mesh | 尚未提供 |
| 2D quadrilateral → 3D hexahedral drag | 支援 |

shared-rail 演算法目前要求 XY feature edges 為水平或垂直。斜線、三角形與一般非正交邊界不在目前演算法範圍內。

`CYLINDER` 的 domain dispatcher 與 `checkerboard_mesh_cylinder()` 入口刻意保留在此 repository，實際 cylinder mesher 可由外部專案提供或在整合時替換。

## 安裝

專案使用 Python 3.9+ 與 `src` layout。

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -e .
```

執行測試：

```bash
python -m pip install pytest
python -m pytest
```

## 快速開始：建立 2D checkerboard mesh

```python
from optimal_checkerboard import OptimalMesh25D
from optimal_checkerboard.data_structure.face import Face
from optimal_checkerboard.data_structure.geometry import Obj
from optimal_checkerboard.data_structure.mesh import Mesh
from optimal_checkerboard.data_structure.metal import Metal


root = Obj("BOX", [0, 0, 24, 18], z=0)
root.add_layer(thk=10, material="SUBSTRATE")

root.metals.append(
    Metal(
        "NORMAL",
        begin=2,
        end=8,
        material="M1",
        ranges=[Face("BOX", [2, 2, 9, 7])],
        holes=[Face("BOX", [13, 10, 19, 15])],
    )
)

root.meshs.append(
    Mesh(
        begin=0,
        end=10,
        line=[[11, 0], [11, 18]],
    )
)

die = Obj("BOX", [15, 3, 22, 9], z=5)
die.add_layer(thk=4, material="DIE")
root.add_child(die)

mesher = OptimalMesh25D()
group_lines_v, group_lines_h, x_list, y_list = mesher.set_pattern_obj(
    root,
    element_size=5.0,
    ratio=0.2,
)

mesh2d = mesher.mesh_checkerboard()
```

主要結果：

```python
mesh2d.nodes       # float array, shape (n_nodes, 3)
mesh2d.elements    # integer array, shape (n_elements, 4)

mesher.x_list      # shared x rail coordinates
mesher.y_list      # shared y rail coordinates
mesher.rails       # shared rail metadata
mesher.get_snap_rules()
mesher.get_restore_rules()
```

`element_size` 是 XY 平面的偏好最大元素尺寸。相鄰 pattern lines 的 rail merge tolerance 為：

```text
merge tolerance = element_size * ratio
```

## Snap 與 restore

shared rail 的預設座標不一定是真實 pattern coordinate。例如：

```text
true feature x = 1.0
true feature x = 1.5
shared rail x  = 1.25
```

若兩條 feature 在 XY span 與 active Z interval 上不衝突，它們可以共用 `x=1.25` 的 checkerboard rail。在對應的 bottom Z：

```python
mesher.apply_snap_rules_at_z(z_value)
```

會將指定 span 上的節點移到 `x=1.0` 或 `x=1.5`。

feature 到達 `top_z` 時，restore rule 會先進入 buffer；下一個更高的 Z event 才會把節點還原到 shared rail coordinate。這使 `top_z` 所在平面仍保留該 feature 的幾何。

建議依序處理所有事件：

```python
z_events = sorted(
    set(mesher.get_snap_rules())
    | set(mesher.get_restore_rules())
)

for z_value in z_events:
    mesher.apply_snap_rules_at_z(z_value)
    layer_nodes = mesh2d.nodes.copy()
    # 使用 layer_nodes 建立或保存目前 Z layer。
```

## 建立 3D mesh

`OptimalMesh25D.build()` 會依 layer stack：

1. 套用目前 Z 的 snap/restore rules。
2. 將 2D elements 分配給 area/material。
3. 沿 Z 方向拉伸。

`Obj` hierarchy 與 `obj_list` 的責任不同：

- `Obj` hierarchy 提供 pattern geometry、absolute placement 與 active Z intervals。
- `obj_list` 提供每個 Z interval 真正要產生的 area/material 與 Z extrusion size。

目前 `build()` 不會自動將 `Obj.layers` 或 `Obj.metals` 轉成 `obj_list`。此外，所有需要套用 snap/restore 的 Z 高度都應出現在 layer stack 中，因為 `build()` 只會在各 layer 的 `z_begin` 呼叫 `apply_snap_rules_at_z()`。

```python
obj_list = [
    [
        {
            "z": 0,
            "element_size": 2,
            "areas": [
                {
                    "type": "BOX",
                    "dim": [0, 0, 24, 18],
                    "material": "SUBSTRATE",
                }
            ],
        },
        {
            "z": 5,
            "element_size": 2,
            "areas": [
                {
                    "type": "BOX",
                    "dim": [0, 0, 24, 18],
                    "material": "SUBSTRATE",
                },
                {
                    "type": "BOX",
                    "dim": [15, 3, 22, 9],
                    "material": "DIE",
                },
            ],
        },
        {
            "z": 9,
            "element_size": 1,
            "areas": [
                {
                    "type": "BOX",
                    "dim": [0, 0, 24, 18],
                    "material": "SUBSTRATE",
                }
            ],
        },
        {
            "z": 10,
        },
    ]
]

dragger = mesher.build(obj_list)
```

有效的 3D 輸出資料為：

```python
nodes = dragger.nodes[:dragger.node_num]
elements = dragger.elements[:dragger.element_num]
element_comps = dragger.element_comps[:dragger.element_num]
components = dragger.comps
```

其 shape 為：

```text
nodes          (n_nodes, 3)   -> [x, y, z]
elements       (n_elems, 8)   -> hexahedral node ids
element_comps  (n_elems,)     -> component/material id
components     dict[str, int] -> material name 到 id 的對照
```

這裡 `build()` layer 中的 `element_size` 是 Z 方向的 extrusion size，不是 `set_pattern_obj()` 使用的 XY element size。

## 使用自訂 2D mesh

如果 2D mesh 由其他 mesher 產生，可以使用：

```python
mesher.set_pattern_obj(root, element_size=5.0, ratio=0.2)
snap_faces = mesher.get_snap_faces()

# 使用外部 mesher 依 snap_faces 建立 custom_mesh。
mesher.mesh_assignment(custom_mesh)
```

`custom_mesh` 必須提供：

```python
custom_mesh.nodes       # shape (n, 2+), floating point
custom_mesh.elements    # shape (m, 4), integer node ids
```

網格必須包含每一條 shared x/y rail，否則 snap rule 無法建立正確的 rail node index。

## 核心術語

- `pattern line`：從 BOX、LINE 或 POLYGON 邊界抽出的水平／垂直線段，並帶有 active Z interval。
- `shared rail`：checkerboard mesh 中的結構 grid line。一條 rail 可以代表多條相近且不衝突的 pattern lines。
- `snap rule`：在 feature 的 `bottom_z`，將一段 rail nodes 移到真實 pattern coordinate 的指令。
- `restore rule`：在 feature 結束後，將 rail nodes 還原到 shared rail coordinate 的指令。
- `span`：feature line 在另一個 XY 軸上的覆蓋範圍。X rail 的 span 是 Y 範圍；Y rail 的 span 是 X 範圍。

## 專案結構

```text
src/optimal_checkerboard/
├── algorithms/       feature extraction、rail grouping、snap、drag
├── data_structure/   Obj、Face、Layer、Metal、Mesh
├── mesh/             2D checkerboard mesh generators 與 domain dispatcher
└── mesher.py         對外的 OptimalMesh25D orchestration API

tests/                pytest regression tests
script/               使用範例與視覺化 client
docs/                 資料格式與演算法文件
```

## 設計重點

- 共享 2D topology，再依 Z layer 移動節點。
- 預處理 rail/node index，避免每次 snap 掃描所有節點。
- 合併 rail 時同時檢查 XY span、active Z interval 與相鄰 rail 順序。
- snap 只改變 node coordinates，不改變 element connectivity。
- 支援外部 2D mesher 與外部 cylinder mesher 的整合。

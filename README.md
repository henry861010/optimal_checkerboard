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
- [效能與容量指引](docs/performance.md)：千萬級 hex 的 array 成本、bounded work sets、
  full-domain capacity reservation 與 benchmark 方法。
- [完整 client 範例](script/client_example.py)：建立 `Obj`、產生 2D mesh、逐層套用 snap rules。
- [3D build 範例](script/example_geometry.py)：指定 layer areas/materials 並產生 3D mesh。

## 支援範圍

| 功能 | 狀態 |
| --- | --- |
| `BOX` pattern face | 支援 |
| 水平／垂直 `LINE` pattern face | 支援 |
| orthogonal `POLYGON` pattern face | 支援 |
| `BOX` root footprint 的 2D mesh | 內建支援 |
| `BOX` footprint 的自訂 2D mesh | 可透過 `mesh_assignment()` 指派並驗證 |
| `CYLINDER` pattern edge | 不進入 shared-rail 演算法 |
| `POLYGON` / `CYLINDER` root footprint 的 generation、custom assignment 與 `build()` | 目前 fail closed (`NotImplementedError`) |
| `BOX` domain 內的 orthogonal `POLYGON` build area | 邊界在整個 slab 可 exact 表示時支援 |
| `CYLINDER` build area | 目前拒絕 |
| 2D quadrilateral → 3D hexahedral drag | 支援 |

shared-rail 演算法目前要求 XY feature edges 為水平或垂直。斜線、三角形與一般非正交邊界不在目前演算法範圍內。

`CYLINDER` 的 low-level dispatcher symbol 仍在 repository 中，但高階 pipeline
尚無法證明其 domain partition 與 build boundary correctness。因此不可藉由外部
`Mesh2D` 繞過驗證；`mesh_assignment()` 與 `build()` 對非 BOX footprint 都會
fail closed。

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

merge tolerance 只是「允許嘗試共用 rail」的上限，不是幾何容差。
pattern coordinates、span endpoints 與 Z events 都保留原始浮點值，不會因為
彼此很接近就被視為同一個 topology coordinate。
即使兩個座標只相差 1 ULP，也仍是不同 rail/target，不會 alias。

在接受共用 rail 之前，mesher 會對每個幾何上不同的 Z active state
進行 snap-plan preflight，檢查 target conflict、corner coupling、rail order 與
quadrilateral convexity。若 optimized plan 不安全，會自動改用每個 pattern
coordinate 各自一條 rail 的 exact-coordinate plan，而不是帶著已知風險繼續生成。
可用 `mesher.rail_optimization_fallback` 查看 fallback 原因；`None` 表示
optimized plan 通過。

`mesher.get_snap_rules()` 與 `mesher.get_restore_rules()` 回傳 deep copies，caller
可修改取得的結果而不影響 mesher 內部 plan。相對地，`mesher.faces`、
`mesher.rails`、`snap_rules_by_z` 或 `restore_rules_by_z` 是已驗證 pattern plan
的一部分，不是可直接編輯的設定介面；直接修改會使 integrity signature 失效，
後續受 integrity 保護的 snap、mesh 或 build 操作會 fail closed；rule map 仍存在時，
getters 在回傳前也會驗證。要變更 pattern，請重新呼叫 `set_pattern_obj()`／
`_set_pattern()`，讓 rails、rules 與 indexes 一起重建並驗證。

## Snap 與 active state restore

shared rail 的預設座標不一定是真實 pattern coordinate。例如：

```text
true feature x = 1.0
true feature x = 1.5
shared rail x  = 1.25
```

若兩條 feature 在 XY span 與 active Z interval 上不衝突，它們可以共用 `x=1.25` 的 checkerboard rail。在任一 Z：

```python
mesher.apply_snap_rules_at_z(z_value)
```

會先將上一個 managed state 改動過的節點恢復為 structural rail
baseline，再一次套用該 Z 的所有 active rules。active interval 為兩端包含：
`z_bottom <= z <= z_top`。因此 `top_z` 平面仍保留 feature，但只要查詢
一個嚴格大於 `top_z` 的 Z，該 feature 不再 active，對應節點就在同一次
transaction 中回到 baseline。執行不依賴遍歷順序；restore 不是另一個
佇列。

Z 值預設精確比對，不會 rounding。一般 layer traversal 應依序處理所有幾何事件：

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

`OptimalMesh25D.build()` 目前要求一個 explicit `BOX` mesh domain，並依 layer stack：

1. 以目前 Z 重建完整 active snap state。
2. 將 2D elements 分配給 area/material。
3. 沿 Z 方向拉伸。

在任何 extrusion 前，`build()` 會對每個 slab 中所有 area outer boundary、
area holes、metal ranges 與 metal holes 做 exact representability preflight。每條
selector edge 必須由 active pattern edge 或未被位移的 structural rail 在整個
`[z_begin, z_end]` 連續表示；只在 slab 部分 Z 存在不夠。此檢查與
topology coordinate comparison 都是 exact。Feature 與 rail spans 都是 closed
interval；若某段 rail 在 slab 中被位移，位移 span 的兩個 endpoints 也視為不可用，
不能拿 baseline rail endpoint 當 selector boundary（除非另有 active pattern 或
domain edge exact 覆蓋該點）。

`Obj` hierarchy 與 `obj_list` 的責任不同：

- `Obj` hierarchy 提供 pattern geometry、absolute placement 與 active Z intervals。
- `obj_list` 提供每個 Z interval 真正要產生的 area/material 與 Z extrusion size。

目前 `build()` 不會自動將 `Obj.layers` 或 `Obj.metals` 轉成 `obj_list`。
每個 object stack 必須嚴格遞增，並且必須在其 Z 範圍內的每個 pattern
`z_bottom` 與 `z_top` 都提供 exact layer boundary。`build()` 會在拉伸前驗證這個
條件；遺漏 event 會直接拋出 `ValueError`，不會讓 extrusion 跨過
pattern 生命週期變化。
最後 sentinel 雖然不產生新 extrusion interval，仍會套用並同步該 Z 的完整
snap state 到最終 3D top plane。

`obj_list` 預設只允許一個 non-empty object stack。多個 stacks 會產生各自的
3D nodes，即使介面接觸也不共點，對 FEM 而言是 non-conformal independent
bodies。只有 caller 明確接受這個 topology 時才可使用：

```python
mesher.build(obj_list, allow_independent_bodies=True)
```

`allow_independent_bodies` 與 `preserve_mesh2d` 都必須是實際的 boolean
（`bool`／`numpy.bool_`）；例如字串 `"False"` 不會被當成 opt-in，而會在任何
mesh mutation 前拒絕。

同一 slab 內的 areas 不得對同一 2D element 重疊主張所有權；有優先區域時必須
以 explicit holes 表達，否則 `build()` 拒絕。Area/range/hole 搜尋以 bounded
chunks 處理，避免建立一份完整 element-coordinate copy。

Material schema 也在 geometry preflight fail closed：每個 area 的 `material`，以及
每個 metal 的 `material` 都必須是 non-empty string；`CONVERT` 另要求 non-empty
`material_o`。Metal `type` 僅接受 exact uppercase `NORMAL`、`CONTINUE`、
`CONVERT`。`NORMAL` 必須提供 numeric、非 bool、有限且位於 `[0, 100]`
（兩端包含）的 `density`。未知／小寫 type、缺漏 label、`NaN`、`inf` 或越界
density 都會在 mesh mutation 前拒絕，不會靜默退回 area base material。
Area `holes`、metal `ranges`／`holes` 若提供，必須是 list/tuple；不接受 iterator
或 generator，以免 preflight 消耗一次後 runtime 得到不同的 geometry。
`EMPTY` 是 Dragger 保留的 component id 0，不可當成 area、metal target 或
`material_o` label；要排除區域請用 explicit holes。若 NORMAL selector 的有限
element areas 加總超出 float64 可表示範圍，build 會 fail closed，不會讓 `inf`
進入 density threshold 而錯配材料。

Z `element_size` 是最大允許間距。每個 slab 在配置 3D output 前會確認
float64 可表示嚴格遞增、不重複且最後精確到達 `z_end` 的 planes。
如果絕對 Z 太大、要求尺寸小於 float64 resolution，或 subdivision/count
溢位，會 fail closed，不會產生 zero-volume hexes。

`preserve_mesh2d=True` 不只保留 node values。若進入 `build()` 前 2D mesh 已在某個
active snap state，該 coordinates、changed-node ids 與 node-buffer ownership 在成功或失敗後
都原樣保留；build 在獨立的一份 working-node copy 上從 baseline 開始。

生成或 `mesh_assignment()` 成功後，mesher 會記錄 authoritative baseline
nodes/connectivity 與 structural indexes。Connectivity 及 structural index arrays
會設為 read-only；nodes 必須維持 writable，因為 snap 是受管理的 coordinate mutation。
每次 `build()` 在 extrusion 前都會驗證 array identity/layout、structural indexes，並以
bounded chunks 計算 baseline nodes/connectivity 的 digest。直接 replacement 或未受管理的
in-place mutation 都會 fail closed；需要換 mesh 時必須重新呼叫
`mesh_assignment()`，不能沿用舊 indexes。
合法 active snap state 另保存 sparse expected targets；build hash baseline 前會先
exact 驗證 tracked nodes 仍是那些 targets，再只在 hash chunks 中代回 baseline。
因此直接修改已 snap 的 rail node 也不會被 baseline restore 遮掉。

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

指派不只檢查某個 rail coordinate 上「有節點」。它會以 exact coordinate
檢查每條 feature span 是否有連續、共線的 quadrilateral-edge chain，以及
每個 mandatory station 是否完整、所有 quadrilaterals 是否嚴格凸且為正面積，
並實際預演各個關鍵 Z state。任一檢查失敗時，assignment 會原子性回滾。

內建 BOX generator 會保留 compact `STRUCTURED_BOX` metadata，並直接由
一維 X/Y axes 算出 rail node ids 與 incident elements。已知為 exact row-major
BOX grid 的 custom mesh 也可將同格式 metadata 放在 `custom_mesh.metadata`，或以
`mesher.mesh_assignment(custom_mesh, structured_metadata=metadata)` 明確傳入。mesher
會逐 chunk exact 檢查 axis values、nodes 與 canonical connectivity，驗證通過後才啟用
direct arithmetic，不會盲目信任 metadata。

未提供 structured metadata 的 custom mesh 會走通用 exact coverage、rail indexes
與 node-to-element CSR adjacency 路徑。若 2D custom mesh 本身就接近千萬節點或元素，
需依 [performance guidance](docs/performance.md) 預留這些驗證 transient 與常駐 index
的額外記憶體。若有提供 metadata 但驗證失敗，assignment 會拒絕，
不會自動降級到通用路徑而隱藏 metadata/topology 不一致。

對 explicit BOX domain，通用 custom path 還會證明每個被 elements 引用的 node 在 domain 內、
邊的 manifold multiplicity 為 1 或 2、所有 unmatched edges 都在完整外邊界上，
且高精度總面積在累計誤差界內等於 BOX 面積。這會拒絕 holes、disconnected regions、
duplicate/non-manifold elements 與 overlapping/crossing embeddings。

## 核心術語

- `pattern line`：從 BOX、LINE 或 POLYGON 邊界抽出的水平／垂直線段，並帶有 active Z interval。
- `shared rail`：checkerboard mesh 中的結構 grid line。一條 rail 可以代表多條相近且不衝突的 pattern lines。
- `snap rule`：描述 feature 的 exact active interval、rail span 與真實 target
  coordinate；套用 Z state 時會使用所有 active rules。
- `restore rule`：以 feature `top_z` 索引的生命週期 metadata；執行時的恢復由
  active-state transaction 自動完成，不是另一個延遲佇列。
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
- 安全 preflight 失敗時自動回退 exact-coordinate rails，pattern correctness 優先於 rail 數量。
- snap 只改變 node coordinates，不改變 element connectivity。
- 支援經完整 partition 驗證的外部 BOX 2D mesher；其他 footprint 目前 fail closed。

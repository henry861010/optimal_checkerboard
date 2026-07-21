# 輸入與輸出資料格式

這份文件定義 `optimal_checkerboard` 對外使用時的資料契約。演算法原理請參考 [演算法與資料流](optimal_checkerboard_algorithm.md)。

## 1. 座標與共同規則

- XY 使用 Cartesian coordinates。
- Z 表示垂直堆疊方向。
- 幾何輸入的數值座標可以是 `int` 或 `float`；2D mesh nodes 會正規化為
  `float64`，避免 `float32` 將很接近但不同的 topology coordinates 壓成同一值。
- pattern feature edge 必須是水平或垂直。
- 所有 Z interval 必須滿足 `bottom_z <= top_z` 或 `begin <= end`。
- XY topology coordinates 與 Z event values 保留原始浮點值，不會 rounding。
- 兩個 topology coordinates 即使只相差 1 ULP 也仍是不同 rail/target，不會
  alias；scale-aware ULP noise 只用來判讀單條 line 的 axis direction。
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

`type` 能表達 POLYGON/CYLINDER 不代表完整 meshing pipeline 已支援該 root
footprint。目前只有 explicit BOX domain 有內建 generation、可驗證的 custom
assignment 與 `build()`。POLYGON 仍可作為 BOX domain 內的 orthogonal pattern/build
selector；CYLINDER 只保留資料模型與 low-level symbol。

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

此 repository 仍有 CYLINDER domain dispatcher symbol，但 CYLINDER 不會被轉成
orthogonal shared-rail pattern lines，且目前沒有可證明其 domain partition 的
high-level pipeline。`mesh_checkerboard()`、帶 CYLINDER footprint 的
`mesh_assignment()` 與 `build()` 皆 fail closed；不可用外部 `Mesh2D` 繞過。

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

若 raw caller 仍要呼叫 `mesh_checkerboard()` 或 `build()`，必須提供 explicit
BOX root mesh domain：

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

`merge_tol` 不會變更 pattern coordinate 的相等關係。preprocessing 會以
geometry-scale snap-plan preflight 檢查每個 exact Z boundary 及相鄰 boundary 之間的
open interval state。若 optimized shared rails 無法保持 target consistency、rail order、
coupled corners 或正面積嚴格凸四邊形，mesher 會自動回退到 `merge_tol=0`
的 exact-coordinate rails。回退原因可由：

```python
mesher.rail_optimization_fallback
```

取得；`None` 表示沒有回退。

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
nodes     shape (n, 2+)  numeric coordinates; assignment normalizes to float64
elements  shape (m, 4)   integer node ids
```

驗證規則：

- `elements` 的 node id 必須位於 `[0, len(nodes))`。
- 每條 feature span 必須在對應 structural rail 上由連續、共線的
  quadrilateral boundary-edge chain 完整覆蓋；只有 span endpoints 或零散 nodes 不足夠。
- 每個 mandatory X/Y station 必須有正長度 mesh edge，且不得從 active
  quadrilateral 內部穿過。
- 上述 coverage 與 rail coordinate 預設使用 exact comparison，不會將彼此接近的 rails
  alias 到同一批 nodes。
- 所有 quadrilaterals 必須是反時針、正面積、嚴格凸四邊形。
- assignment 會實際套用每個拓撲上不同的 critical Z snap state，只接受所有
  state 都保持有效的 mesh。
- snapping 可以改變 coordinates，但不應改變 topology 或 node ids。

任一檢查失敗時，`mesh_assignment()` 會還原 caller 的 `nodes`/`elements`
attributes 與 mesher 先前的 mesh state，不留下半更新的 assignment。

外部 mesher 可先取得 shared-rail default faces：

```python
snap_faces = mesher.get_snap_faces()
```

這個方法回傳 deep copies，不會修改原始 `mesher.faces`。

內建 BOX generator 會產生 `STRUCTURED_BOX` metadata（`grid_shape`、`x_nodes`、
`y_nodes`）供 mesher 直接以一維 axis 算出 rail node ids 與 incident elements。
已知為 exact row-major BOX grid 的 custom mesh 可以二擇一提供同格式 metadata：

```python
mesh2d.metadata = metadata
mesher.mesh_assignment(mesh2d)

# 或明確傳入：
mesher.mesh_assignment(mesh2d, structured_metadata=metadata)
```

若兩者同時存在，明確傳入的 `structured_metadata` 優先。

```python
metadata = {
    "kind": "STRUCTURED_BOX",
    "grid_shape": (len(y_nodes), len(x_nodes)),
    "x_nodes": x_nodes,
    "y_nodes": y_nodes,
}
```

metadata 不是跳過 safety checks 的宣告。mesher 會逐 chunk exact 驗證
`grid_shape`、strictly increasing axis values、每個 row-major node coordinate 與每個
canonical quadrilateral connectivity，成功後才改用 direct axis/incident-element
arithmetic。

沒有提供 metadata 的 custom mesh 會走通用 exact edge-chain/station coverage、
rail indexes 與 node-to-element CSR adjacency。對接近千萬級的 2D custom mesh，
請事先依 [Performance and capacity guidance](performance.md) 估算驗證 transient
與常駐 index 記憶體。已提供但驗證失敗的 metadata 會使 assignment
失敗，不會自動改走通用路徑。

當 pattern 有 explicit BOX `mesh_domain` 時，通用 custom path 還會對整個
domain partition 做 fail-closed proof：

- 每個 `elements` 引用的 node 都必須在 BOX 內；
- 每條 undirected edge 的 multiplicity 必須為 1 或 2，拒絕 duplicate elements
  與 non-manifold edges；
- multiplicity-one edges 必須全部位於 physical BOX boundary，並連續覆蓋四邊；
- 以 `longdouble` 累計的總面積必須在只由累計次數決定的誤差界內
  等於 domain 面積。

因此 holes、disconnected regions、internal unmatched edges 與 overlapping/crossing
embeddings 都會被拒絕。驗證過的 `STRUCTURED_BOX` path 則以 exact
row-major topology 與 axis endpoints 等於 BOX bounds 作為 compact proof。

目前 explicit `POLYGON` 或 `CYLINDER` footprint 的 custom domain partition 尚未實作，
`mesh_assignment()` 會拋出 `NotImplementedError`。沒有 `mesh_domain` 的 private raw-face
caller 仍可指派 2D mesh 來執行 snap API，但因為無法證明完整 footprint，
不能呼叫 `build()`。

## 6. `build()` 的 3D layer stack

```python
dragger = mesher.build(
    obj_list,
    preserve_mesh2d=False,
    allow_independent_bodies=False,
)
```

`build()` 需要 explicit `BOX` mesh domain，才能證明完整 footprint coverage。
沒有 domain 時拋出 `RuntimeError`；`POLYGON` 或 `CYLINDER` footprint 則因為
partition proof 尚未實作而拋出 `NotImplementedError`。

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

預設最多只能有一個至少兩筆 entries 的 non-empty stack。多 stack 並不
建立 conformal interface；每個 stack 會產生獨立 3D nodes，即使座標接觸也不共點。
只有當 caller 明確要 independent FEM bodies 時才可選擇：

```python
mesher.build(obj_list, allow_independent_bodies=True)
```

`allow_independent_bodies` 與 `preserve_mesh2d` 的值必須是 `bool` 或
`numpy.bool_`。其他 truthy/falsy 值（例如字串 `"False"`、整數 `1`）都會在
任何 mesh mutation／3D allocation 前拒絕，不能意外開啟 independent-body
topology 或 preservation mode。

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

每個 object stack 的 Z 必須是有限值且嚴格遞增。`build()` 只在每個非
sentinel entry 的 `z` 重建 active snap state，因此 stack 不得跨過任何沒有
layer boundary 的 pattern lifecycle event。對每個位於 stack 起訖 Z 之間的 exact
`z_bottom` 與 `z_top`，stack 都必須有完全相同的 Z entry。這是強制驗證：
遺漏 event 會在產生任何 3D output 前拋出 `ValueError`。

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

同一個 object stack 具有持續的 2D material state。第一個 layer 從空狀態開始；
之後每個 layer 的 `areas` 都是 overlay patches：被 area 選到的 elements 依該 area
重新分配，未被任何 area 選到的 elements 則繼承上一個 interval 的 material。
因此較小的 child area 可以覆寫 main 的局部，同時 main 的其餘 footprint 會繼續
拉伸。只有開始另一個 object stack 時才會重設 material state，避免不同 bodies
互相繼承。

### 6.3 Final sentinel

```python
{
    "z": final_z,
}
```

即使 sentinel 含有其他欄位，`build()` 也不會處理其 `areas`。但 sentinel
不是被忽略的幾何平面：它的 exact Z active snap state 會被套用並同步到
已生成的最終 3D top plane，防止 top surface 留在前一個 rail state。

### 6.4 Area

在任何 3D allocation 或 mesh mutation 前，`build()` 會對每個
`[z_begin, z_end]` slab 中的下列邊界做 exact representability preflight：

1. 每個 area 的 outer boundary。
2. 每個 area hole。
3. 每個 area metal 的 `ranges` 與 `holes`。

每條 selector edge 必須由 original active pattern edges、永久 BOX domain edges，或該
slab 內完全沒有被 snap 位移的 structural rail intervals 連續覆蓋。
Pattern edge 的 active Z 必須包住整個 slab；只在一個 endpoint 或 slab 部分高度
存在都不算可表示。座標、span union 與 Z coverage 都是 exact comparison。

Rail 與 feature spans 採 closed interval。若 baseline rail 的 `[span_min, span_max]`
有一段在 slab 內被 snap rule 位移，該 displaced span 的兩個 endpoints 也從
baseline 可用範圍排除；實作以 `nextafter` 表達兩側仍可使用的最近 float64
座標。因此 selector boundary 不能只靠 displaced endpoint 冒充未位移 rail；只有
另一條 active pattern edge 或永久 domain edge exact 覆蓋時才可能通過。

```python
{
    "type": "BOX" | "POLYGON",
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

每個 area 的 `material` 是必要欄位，且必須是 non-empty string。缺漏、空字串或
非字串會在 build geometry preflight 拒絕，不會建立匿名/default component。

Area selection 以完整 2D quadrilateral element 為單位；元素四個角都位於 area 內時，該元素才會被選取。

`POLYGON` area 可用於 explicit BOX domain 內，但必須是 orthogonal 且通過上述
slab-boundary proof。`CYLINDER` selector 目前無法由 orthogonal pattern edges exact
表示，high-level `build()` 會拒絕。

同一 slab 內兩個 areas 不得同時包含同一 2D element。有 priority region 時要用
explicit `holes` 從其中一個 area 排除；否則拋出 `ValueError`，而不以
list order 靜默覆寫 material。Area/range/hole 搜尋將 2D elements 分成 bounded
chunks，不會配置一份與整個 mesh 同大的 element-coordinate copy。

`holes` 使用與 area 相同的 face dictionary：

```python
"holes": [
    {
        "type": "BOX",
        "dim": [4, 4, 6, 6],
    }
]
```

Area `holes` 若提供，必須是 list/tuple，且每一項都是 selector mapping。
Iterator/generator 會被拒絕；這避免同一份輸入在 geometry preflight 被消耗後，
runtime classification 看見不同或空的 holes。

Hole 的語意是「這個 area patch 不覆寫該處」。若 hole 內在上一個 interval 已有
material，該 material 會照常繼承；hole 不會隱式清除既有 footprint。

### 6.5 Area metals

這裡的 metal dictionaries 屬於 3D material assignment schema，與 `data_structure.Metal` 是不同層次的資料。

`metals` 若提供，必須是 list/tuple，且每一項都是 mapping。共通 schema 為：

| Metal type | 必要欄位 |
| --- | --- |
| `NORMAL` | non-empty string `material`；numeric、非 bool、finite 且在 `[0, 100]` 內的 `density` |
| `CONTINUE` | non-empty string `material` |
| `CONVERT` | non-empty string `material` 與 `material_o` |

`type` 是 case-sensitive discriminator，只接受 exact uppercase `NORMAL`、
`CONTINUE`、`CONVERT`。未知或 lowercase type、缺少 material/material_o、
`NORMAL` 缺少或提供 non-numeric density，以及 `NaN`、`inf`、負值或大於 `100`
的 density 都會在
任何 mesh mutation／3D allocation 前 fail closed。它們不會被忽略，也不會讓
未匹配 elements 靜默退回 area base material。

每個 metal 的 `ranges` 與 `holes` 若提供，也必須是 list/tuple，且每一項都是
selector mapping；不接受一次性 iterator/generator。

`EMPTY` 是內部保留的 component label（id `0`），代表不產生 element。它不可作為
area `material`、任何 metal target `material`，也不可作為 `CONVERT.material_o`。
需要讓 area 不命中某區域時請使用 explicit holes，而不是用 reserved label；但在
overlay layer 中，hole 只保留該處原有 material，不會刪除已存在的 elements。

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

`density` 是目標 area volume 的百分比，允許 `0` 與 `100`。選取順序使用固定
seed，因此相同 layer/input 應可重現。
所有 individual quad areas 與 selector area sum 都必須可有限表示。若正值有限的
individual areas 加總超出 float64 range，build 會拋出 `OverflowError`；不會以
`inf` 計算 density threshold 後回傳錯誤材料比例。

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

`build()` 只複製一份 working nodes 給 `Dragger`，並在這份 copy 上從
structural baseline 開始 build。原始 `mesher.mesh2d.nodes` 不會被改動。若進入
`build()` 之前原 mesh 已套用某個 active snap state，不只 coordinates，連同
sparse changed-node ids 與 `_snap_state_nodes` buffer ownership 都會在成功或失敗後
原樣保留。因此 build 之後可在原 buffer 上繼續原 active-state workflow。

### 6.7 Z subdivision representability

layer `element_size` 是 Z 方向元素的最大尺寸，不只是偏好值。每個
slab 在修改 node mappings 或配置 output 前，會分 chunk 驗證所有 float64
Z planes：

- `begin < end`，且所有 planes 有限、嚴格遞增、不重複；
- 最後 plane 精確等於 `end`；
- 每個 interval 不超過 requested `element_size` 加上最多數個可表示
  float64 steps 的算術 allowance；topology order 本身沒有容差；
- output count 不得超過 int32 ids/connectivity capacity。
- simple full-domain capacity reservation 也必須先完成上述所有 Z-plane proof，
  才能配置 final arrays。

若絕對 Z 太大使 `element_size` 小於 float64 resolution、subdivision ratio 溢位，
或任一 planes 會重合/反轉，`build()` 會 fail closed，不會產生 zero-volume
hexahedra。

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

兩個 getters 都回傳 deep copies，包含內層 rule dictionaries；修改回傳值不會改動
mesher 的 validated plan。`mesher.faces`、`mesher.rails`、
`mesher.snap_rules_by_z` 與 `mesher.restore_rules_by_z` 則應視為 read-only by
contract。直接修改這些 public objects 不會成為合法設定，而會使 pattern integrity
signature 不一致；後續受 integrity 保護的 snap、mesh 或 build 操作會拒絕，
rule map 仍存在時 getters 在回傳前也會驗證。要變更幾何或 rules，必須重新執行
`set_pattern_obj()`／`_set_pattern()`。

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

`restore_rules_by_z` 是以 feature top Z 索引的生命週期 metadata，可用於列舉完整
layer boundaries。`apply_snap_rules_at_z()` 不會將它放入另一個執行佇列；實際執行是
先恢復上一 managed state 曾改動的 nodes，再套用查詢 Z 的所有 active
snap rules。

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

active condition 為兩端包含：

```text
z_bottom <= z <= z_top
```

所以查詢 `top_z` 時 feature 仍在；查詢任一嚴格大於 `top_z` 的 Z 時，
它就不再 active 並回到 structural baseline。結果只由查詢 Z 決定，不依賴
之前呼叫的 Z 順序。`eps` 預設為 `0.0`，代表 exact event lookup；若 caller
明確傳入正容差，它只用來解析查詢 Z 到唯一相鄰 event，不會合併或
改寫內部 event keys。同時等距於兩個 events 會被拒絕為 ambiguous。

### 7.5 `Mesh2D`

```python
mesh2d = mesher.mesh_checkerboard()
```

```text
mesh2d.nodes     shape (n_nodes, 3), float64
mesh2d.elements  shape (n_elements, 4), int32
mesh2d.metadata  optional structured topology mapping
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

內建 BOX generator 的 `mesh2d.metadata` 為已驗證的 `STRUCTURED_BOX` mapping。
若 caller 將 mesh arrays 或 axes 另作修改，不可沿用過時 metadata；之後再以
`mesh_assignment()` 指派時，mesher 會重新 exact 驗證它與 arrays 是否一致。

成功 generation／assignment 後的 ownership contract：

- `mesh2d.nodes` 保持 writable，僅供 `apply_snap_rules_at_z()` 與
  `reset_snap_state()` 做 managed coordinate changes。
- `mesh2d.elements` 及 rail/node、node/element adjacency 等 structural index
  arrays 會設成 read-only，普通 in-place mutation 會立即失敗。
- mesher 記錄 nodes/elements 的 authoritative identity、layout 與 baseline
  digest，也記錄 structural index layout/digest。`build()` 會先回到可驗證的
  baseline，再用 bounded chunks hash nodes/connectivity；array replacement、強制
  解鎖後的 mutation，或未受管理的 node edit 都會 fail closed。
- active snap state 同時記錄 changed ids 與 exact expected target values；baseline
  digest 只會虛擬代回已驗證的 managed values。直接修改一個已 snap node 不會被
  restore 動作遮蔽，仍會在任何 3D allocation 前拒絕。

需要編輯 custom mesh 時，請複製 arrays、完成編輯後重新呼叫
`mesh_assignment()`。不要替換已建立 index 的 `mesh2d.nodes`／`elements`，也不要
直接改 `rail_node_index` 等 internal structural mappings。

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

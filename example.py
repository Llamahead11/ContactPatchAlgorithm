import open3d as o3d

dtype_f = o3d.core.float32
dtype_i = o3d.core.int32

# Create an empty line set
# Use lineset.point to access the point attributes
# Use lineset.line to access the line attributes
lineset = o3d.t.geometry.LineSet()

# Default attribute: point.positions, line.indices
# These attributes is created by default and are required by all line
# sets. The shape must be (N, 3) and (N, 2) respectively. The device of
# "positions" determines the device of the line set.
lineset.point.positions = o3d.core.Tensor([[0, 0, 0],
                                              [0, 0, 1]], dtype_f)
lineset.line.indices = o3d.core.Tensor([[0, 1]], dtype_i)

# Common attributes: line.colors
# Common attributes are used in built-in line set operations. The
# spellings must be correct. For example, if "color" is used instead of
# "color", some internal operations that expects "colors" will not work.
# "colors" must have shape (N, 3) and must be on the same device as the
# line set.
lineset.line.colors = o3d.core.Tensor([[0.0, 0.0, 0.0]], dtype_f)

# User-defined attributes
# You can also attach custom attributes. The value tensor must be on the
# same device as the line set. The are no restrictions on the shape or
# dtype, e.g.,
#lineset.point.labels = o3d.core.Tensor(...)
#lineset.line.features = o3d.core.Tensor(...)

o3d.visualization.draw_geometries([lineset.to_legacy()])
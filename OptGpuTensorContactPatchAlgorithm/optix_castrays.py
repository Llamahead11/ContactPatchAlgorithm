# optix_raycaster.py - Reusable OptiX raycaster class for real-time GPU ray sampling

import open3d as o3d
import cupy as cp
import numpy as np
import os
import optix
from cuda.bindings import runtime, nvrtc
import ctypes

import path_util

def checkNVRTC(result, prog = None):
    if result[0].value:
        if prog:
            (res, logsize) = nvrtc.nvrtcGetProgramLogSize(prog)
            if not res.value:
                log = b" " * logsize
                nvrtc.nvrtcGetProgramLog(prog, log)
                print(log.decode())
        raise RuntimeError("NVRTC error code={}({})".format(result[0].value, nvrtc.nvrtcGetErrorString(result[0])[1]))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]

class Logger:
    def __init__( self ):
        self.num_mssgs = 0

    def __call__( self, level, tag, mssg ):
        print( "[{:>2}][{:>12}]: {}".format( level, tag, mssg ) )
        self.num_mssgs += 1

def log_callback( level, tag, mssg ):
    print( "[{:>2}][{:>12}]: {}".format( level, tag, mssg ) )
    

def round_up( val, mult_of ):
    return val if val % mult_of == 0 else val + mult_of - val % mult_of 

def get_aligned_itemsize( formats, alignment ):
    names = []
    for i in range( len(formats ) ):
        names.append( 'x'+str(i) )

    temp_dtype = np.dtype( { 
        'names'   : names,
        'formats' : formats, 
        'align'   : True
        } )
    return round_up( temp_dtype.itemsize, alignment )


def optix_version_gte( version ):
    if optix.version()[0] >  version[0]:
        return True
    if optix.version()[0] == version[0] and optix.version()[1] >= version[1]:
        return True
    return False


def array_to_device_memory( numpy_array, stream=cp.cuda.Stream() ):

    byte_size = numpy_array.size*numpy_array.dtype.itemsize

    h_ptr = ctypes.c_void_p( numpy_array.ctypes.data )
    d_mem = cp.cuda.memory.alloc( byte_size )
    d_mem.copy_from_async( h_ptr, byte_size, stream )
    return d_mem

def compile_cuda( cuda_file ):
    compile_options = [
        b'-use_fast_math', 
        b'-lineinfo',
        b'-default-device',
        b'-std=c++11',
        b'-rdc',
        b'true',
        f'-I{path_util.include_path}'.encode(),
        f'-I{path_util.cuda_tk_path}'.encode(),
    ]
    # Optix 7.0 compiles need path to system stddef.h
    # the value of optix.stddef_path is compiled in constant. When building
    # the module, the value can be specified via an environment variable, e.g.
    #   export PYOPTIX_STDDEF_DIR="/usr/include/linux"
    if not optix_version_gte( (7,1) ):
        compile_options.append( f'-I{path_util.stddef_path}' )
    print("pynvrtc compile options = {}".format(compile_options))

    with open( cuda_file, 'rb' ) as f:
        src = f.read()

    # Create program
    prog = checkNVRTC(nvrtc.nvrtcCreateProgram(src, cuda_file.encode(), 0, [], []))

    # Compile program
    checkNVRTC(nvrtc.nvrtcCompileProgram(prog, len(compile_options), compile_options), prog)

    # Get PTX from compilation
    ptxSize = checkNVRTC(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b" " * ptxSize
    checkNVRTC(nvrtc.nvrtcGetPTX(prog, ptx))
    return ptx


class OptiXRaycaster:
    def __init__(self, ply_path_inner, ply_path_outer, ply_path_tread, cuda_file, stream):
        cp.cuda.runtime.free( 0 )
        triangle_cu = os.path.join(os.path.dirname(__file__), 'raycast.cu')
        self.triangle_ptx = compile_cuda( triangle_cu )

        #self.init_optix()
        #cp.cuda.runtime.free( 0 )
        self.stream = stream
        self.ctx = self.create_ctx()
        self.vertices_tread, self.indices_tread = self._load_mesh_to_gpu_1(ply_path_tread)
        self.tread_gas_handle, d_gas_output_tread = self._build_gas_1(self.ctx, self.vertices_tread, self.indices_tread)
        cp.cuda.runtime.deviceSynchronize()
        self.vertices_outer, self.indices_outer = self._load_mesh_to_gpu_1(ply_path_outer)
        self.outer_gas_handle, d_gas_output_outer = self._build_gas_1(self.ctx, self.vertices_outer, self.indices_outer)
        cp.cuda.runtime.deviceSynchronize()
        self.vertices_inner, self.indices_inner, self.triangle_normals_inner = self._load_mesh_to_gpu(ply_path_inner)
        self.inner_gas_handle, d_gas_output_inner = self._build_gas(self.ctx, self.vertices_inner, self.indices_inner)
        cp.cuda.runtime.deviceSynchronize()
        print("GAS 0 vertices ptr:", int(self.vertices_outer.data.ptr))
        print("GAS 1 vertices ptr:", int(self.vertices_inner.data.ptr))
        print("GAS 0 indices ptr:", int(self.indices_outer.data.ptr))
        print("GAS 1 indices ptr:", int(self.indices_inner.data.ptr))
        self.pipeline_options = self.set_pipeline_options()
        self.module = self.create_module()
        self.prog_groups = self.create_program_groups()
        self.pipeline = self.create_pipeline()
        self.sbt = self.create_sbt()

        # self.ptx = self._compile_cuda(cuda_file)
        # self.pipeline, self.sbt = self._create_pipeline(self.ptx)

    def init_optix(self):
        print( "Initializing cuda ..." )
        cp.cuda.runtime.free( 0 )

        print( "Initializing optix ..." )
        optix.init()

    def create_ctx(self):
        print( "Creating optix device context ..." )

        # Note that log callback data is no longer needed.  We can
        # instead send a callable class instance as the log-function
        # which stores any data needed
        global logger
        logger = Logger()
        
        # OptiX param struct fields can be set with optional
        # keyword constructor arguments.
        ctx_options = optix.DeviceContextOptions( 
                logCallbackFunction = logger,
                logCallbackLevel    = 4
                )

        # They can also be set and queried as properties on the struct
        if optix.version()[1] >= 2:
            print(optix.version())
            ctx_options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL 

        cu_ctx = 0 
        return optix.deviceContextCreate( cu_ctx, ctx_options )

    def _load_mesh_to_gpu(self, ply_path):
        # pcd = o3d.t.io.read_point_cloud(ply_path)
        # pcd.scale(scale = 0.03912, center = [0,0,0])
        # pcd.estimate_normals()
        # pcd.orient_normals_consistent_tangent_plane(k = 10)
        normals = o3d.core.Tensor.load('inner_normals_oriented.npy')
        mesh = o3d.t.io.read_triangle_mesh(ply_path)
        mesh.scale(scale = 0.03912, center = [0,0,0])
        mesh.vertex.normals = normals #pcd.point.normals
        #mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh.normalize_normals()
        with self.stream:
            vertices = cp.asarray(mesh.vertex.positions.numpy(), dtype=np.float32)
            indices = cp.asarray(mesh.triangle.indices.numpy(), dtype=np.uint32)
            normals = cp.asarray(mesh.triangle.normals.numpy(), dtype=np.float32)
            vertices = cp.ascontiguousarray(vertices)  # shape (N,3)
            indices = cp.ascontiguousarray(indices)
            normals = cp.ascontiguousarray(normals)
        print(cp.sum(cp.isnan(vertices).any()), cp.sum(cp.isinf(vertices).any())) ### KEEP IN!!! FORCE SYNCHRONIZE!!!
        # assert not cp.isnan(vertices).any(), "NaNs in vertex array!"
        # assert not cp.isinf(vertices).any(), "Infs in vertex array!"
        # print(vertices.shape, vertices.strides, indices.shape, indices.strides)
        # print("Vertices contiguous:", vertices.flags['C_CONTIGUOUS'])
        # print("Indices contiguous:", indices.flags['C_CONTIGUOUS'])
        # print(vertices.dtype, indices.dtype)
        self.stream.synchronize() 
        
        return vertices, indices, normals
    
    def _load_mesh_to_gpu_1(self, ply_path):
        mesh = o3d.t.io.read_triangle_mesh(ply_path)
        mesh.scale(scale = 0.03912, center = [0,0,0])
        #mesh = mesh.filter_smooth_simple(10)
        mesh.compute_vertex_normals()
        with self.stream:
            vertices = cp.asarray(mesh.vertex.positions.numpy(), dtype=np.float32)
            indices = cp.asarray(mesh.triangle.indices.numpy(), dtype=np.uint32)
            vertices = cp.ascontiguousarray(vertices)  # shape (N,3)
            indices = cp.ascontiguousarray(indices)
        print(cp.sum(cp.isnan(vertices).any()), cp.sum(cp.isinf(vertices).any())) ### KEEP IN!!! FORCE SYNCHRONIZE!!!
        # assert not cp.isnan(vertices).any(), "NaNs in vertex array!"
        # assert not cp.isinf(vertices).any(), "Infs in vertex array!"
        # print(vertices.shape, vertices.strides, indices.shape, indices.strides)
        # print("Vertices contiguous:", vertices.flags['C_CONTIGUOUS'])
        # print("Indices contiguous:", indices.flags['C_CONTIGUOUS'])
        # print(vertices.dtype, indices.dtype)
        self.stream.synchronize() 
        
        return vertices, indices
    
    def _build_gas(self,ctx,vertices,indices):
        
        accel_options = optix.AccelBuildOptions(
            buildFlags = int( optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS),
            operation  = optix.BUILD_OPERATION_BUILD
            )
            
        triangle_input_flags = [ optix.GEOMETRY_FLAG_NONE ]
        triangle_input = optix.BuildInputTriangleArray()
        triangle_input.vertexFormat  = optix.VERTEX_FORMAT_FLOAT3
        triangle_input.vertexStrideInBytes = 12
        triangle_input.numVertices   = vertices.shape[0]
        triangle_input.vertexBuffers = [ int(vertices.data.ptr) ]

        triangle_input.indexFormat         = optix.INDICES_FORMAT_UNSIGNED_INT3
        triangle_input.indexStrideInBytes  = 12  # 3 * sizeof(uint32)
        triangle_input.numIndexTriplets    = indices.shape[0]
        triangle_input.indexBuffer         = int(indices.data.ptr)

        triangle_input.flags         = triangle_input_flags
        triangle_input.numSbtRecords = 1;
            
        gas_buffer_sizes = ctx.accelComputeMemoryUsage( [accel_options], [triangle_input] )

        d_temp_buffer_gas   = cp.cuda.alloc( gas_buffer_sizes.tempSizeInBytes )
        d_gas_output_buffer = cp.cuda.alloc( gas_buffer_sizes.outputSizeInBytes)
        
        gas_handle = ctx.accelBuild( 
            self.stream.ptr,    # CUDA stream
            [ accel_options ],
            [ triangle_input ],
            d_temp_buffer_gas.ptr,
            gas_buffer_sizes.tempSizeInBytes,
            d_gas_output_buffer.ptr,
            gas_buffer_sizes.outputSizeInBytes,
            [] # emitted properties
            )
        print("Gas handle: ", gas_handle)

        # print(f'vertices device ptr: {int(vertices.data.ptr):#x}')
        # print(f'indices device ptr: {int(indices.data.ptr):#x}')
        # print(f'vertices shape: {vertices.shape}, strides: {vertices.strides}')
        # print(len(vertices)//3, len(indices)//3)
        # print(f'indices shape: {indices.shape}, strides: {indices.strides}')
        # print(f'vertices dtype: {vertices.dtype}, indices dtype: {indices.dtype}')
        # print(f'vertices is contiguous: {vertices.flags.c_contiguous}')
        # print(f'indices is contiguous: {indices.flags.c_contiguous}')
        self.stream.synchronize()
        return gas_handle, d_gas_output_buffer
    
    def _build_gas_1(self,ctx,vertices,indices):
        
        accel_options = optix.AccelBuildOptions(
            buildFlags = int( optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS),
            operation  = optix.BUILD_OPERATION_BUILD
            )
            
        triangle_input_flags = [ optix.GEOMETRY_FLAG_NONE ]
        triangle_input = optix.BuildInputTriangleArray()
        triangle_input.vertexFormat  = optix.VERTEX_FORMAT_FLOAT3
        triangle_input.vertexStrideInBytes = 12
        triangle_input.numVertices   = vertices.shape[0]
        triangle_input.vertexBuffers = [ int(vertices.data.ptr) ]

        triangle_input.indexFormat         = optix.INDICES_FORMAT_UNSIGNED_INT3
        triangle_input.indexStrideInBytes  = 12  # 3 * sizeof(uint32)
        triangle_input.numIndexTriplets    = indices.shape[0]
        triangle_input.indexBuffer         = int(indices.data.ptr)

        triangle_input.flags         = triangle_input_flags
        triangle_input.numSbtRecords = 1;
            
        gas_buffer_sizes = ctx.accelComputeMemoryUsage( [accel_options], [triangle_input] )

        d_temp_buffer_gas   = cp.cuda.alloc( gas_buffer_sizes.tempSizeInBytes )
        d_gas_output_buffer = cp.cuda.alloc( gas_buffer_sizes.outputSizeInBytes)
        
        gas_handle = ctx.accelBuild( 
            self.stream.ptr,    # CUDA stream
            [ accel_options ],
            [ triangle_input ],
            d_temp_buffer_gas.ptr,
            gas_buffer_sizes.tempSizeInBytes,
            d_gas_output_buffer.ptr,
            gas_buffer_sizes.outputSizeInBytes,
            [] # emitted properties
            )
        print("Gas handle: ", gas_handle)

        # print(f'vertices device ptr: {int(vertices.data.ptr):#x}')
        # print(f'indices device ptr: {int(indices.data.ptr):#x}')
        # print(f'vertices shape: {vertices.shape}, strides: {vertices.strides}')
        # print(len(vertices)//3, len(indices)//3)
        # print(f'indices shape: {indices.shape}, strides: {indices.strides}')
        # print(f'vertices dtype: {vertices.dtype}, indices dtype: {indices.dtype}')
        # print(f'vertices is contiguous: {vertices.flags.c_contiguous}')
        # print(f'indices is contiguous: {indices.flags.c_contiguous}')
        self.stream.synchronize()
        return gas_handle, d_gas_output_buffer
    
    def set_pipeline_options(self):
        if optix.version()[1] >= 2:
            return optix.PipelineCompileOptions(
                usesMotionBlur         = False,
                traversableGraphFlags  = int( optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY ),
                numPayloadValues       = 15,
                numAttributeValues     = 4,
                exceptionFlags         = int( optix.EXCEPTION_FLAG_NONE ),
                pipelineLaunchParamsVariableName = "params",
                usesPrimitiveTypeFlags = optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE
            )
        else:
            return optix.PipelineCompileOptions(
                usesMotionBlur         = False,
                traversableGraphFlags  = int( optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY ),
                numPayloadValues       = 15,
                numAttributeValues     = 4,
                exceptionFlags         = int( optix.EXCEPTION_FLAG_NONE ),
                pipelineLaunchParamsVariableName = "params"
            )

    def create_module(self):
        print( "Creating optix module ..." )
        

        module_options = optix.ModuleCompileOptions(
            maxRegisterCount = optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            optLevel         = optix.COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel       = optix.COMPILE_DEBUG_LEVEL_DEFAULT
        )

        module, log = self.ctx.moduleCreate(
                module_options,
                self.pipeline_options,
                self.triangle_ptx
                )
        print( "\tModule create log: <<<{}>>>".format( log ) )
        return module

    def create_program_groups(self):
        print( "Creating program groups ... " )

        raygen_prog_group_desc                          = optix.ProgramGroupDesc()
        raygen_prog_group_desc.raygenModule             = self.module
        raygen_prog_group_desc.raygenEntryFunctionName  = "__raygen__rg"
        raygen_prog_group, log = self.ctx.programGroupCreate(
            [ raygen_prog_group_desc ]
            )
        print( "\tProgramGroup raygen create log: <<<{}>>>".format( log ) )
        
        miss_prog_group_desc                        = optix.ProgramGroupDesc()
        miss_prog_group_desc.missModule             = self.module
        miss_prog_group_desc.missEntryFunctionName  = "__miss__ms"
        program_group_options = optix.ProgramGroupOptions() 
        miss_prog_group, log = self.ctx.programGroupCreate(
            [ miss_prog_group_desc ]
            )
        print( "\tProgramGroup miss create log: <<<{}>>>".format( log ) )

        hitgroup_prog_group_desc                             = optix.ProgramGroupDesc()
        hitgroup_prog_group_desc.hitgroupModuleCH            = self.module
        hitgroup_prog_group_desc.hitgroupEntryFunctionNameCH = "__closesthit__ch"
        hitgroup_prog_group, log = self.ctx.programGroupCreate(
            [ hitgroup_prog_group_desc ]
            )
        print( "\tProgramGroup hitgroup create log: <<<{}>>>".format( log ) )

        return [ raygen_prog_group[0], miss_prog_group[0], hitgroup_prog_group[0] ]

    def create_pipeline(self):
        print( "Creating pipeline ... " )

        max_trace_depth  = 1
        pipeline_link_options               = optix.PipelineLinkOptions() 
        pipeline_link_options.maxTraceDepth = max_trace_depth

        log = ""
        pipeline = self.ctx.pipelineCreate(
                self.pipeline_options,
                pipeline_link_options,
                self.prog_groups,
                log)

        stack_sizes = optix.StackSizes()
        for prog_group in self.prog_groups:
            if optix_version_gte( (7,7) ):
                optix.util.accumulateStackSizes( prog_group, stack_sizes, pipeline )
            else: 
                optix.util.accumulateStackSizes( prog_group, stack_sizes )

        (dc_stack_size_from_trav, dc_stack_size_from_state, cc_stack_size) = \
            optix.util.computeStackSizes( 
                stack_sizes, 
                max_trace_depth,
                0,  # maxCCDepth
                0   # maxDCDepth
                )
        
        pipeline.setStackSize( 
                dc_stack_size_from_trav,
                dc_stack_size_from_state, 
                cc_stack_size,
                1  # maxTraversableDepth
                )

        return pipeline

    def create_sbt(self):
        print( "Creating sbt ... " )

        (raygen_prog_group, miss_prog_group, hitgroup_prog_group ) = self.prog_groups

        global d_raygen_sbt
        global d_miss_sbt

        header_format = '{}B'.format( optix.SBT_RECORD_HEADER_SIZE )

        #
        # raygen record
        #
        formats  = [ header_format ]
        itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
        dtype = np.dtype( { 
            'names'   : ['header' ],
            'formats' : formats, 
            'itemsize': itemsize,
            'align'   : True
            } )
        h_raygen_sbt = np.array( [ 0 ], dtype=dtype )
        optix.sbtRecordPackHeader( raygen_prog_group, h_raygen_sbt )
        global d_raygen_sbt 
        d_raygen_sbt = array_to_device_memory( h_raygen_sbt , self.stream)
        
        #
        # miss record
        #
        formats  = [ header_format]
        itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
        dtype = np.dtype( { 
            'names'   : ['header'],
            'formats' : formats,
            'itemsize': itemsize,
            'align'   : True
            } )
        h_miss_sbt = np.array( [ 0 ], dtype=dtype )
        optix.sbtRecordPackHeader( miss_prog_group, h_miss_sbt )
        global d_miss_sbt 
        d_miss_sbt = array_to_device_memory( h_miss_sbt , self.stream)
        
        #
        # hitgroup record
        #
        formats  = [ header_format ]
        itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
        dtype = np.dtype( { 
            'names'   : ['header' ],
            'formats' : formats,
            'itemsize': itemsize,
            'align'   : True
            } )
        h_hitgroup_sbt = np.array( [ (0) ], dtype=dtype )
        optix.sbtRecordPackHeader( hitgroup_prog_group, h_hitgroup_sbt )
        global d_hitgroup_sbt
        d_hitgroup_sbt = array_to_device_memory( h_hitgroup_sbt , self.stream)
        
        return optix.ShaderBindingTable(
            raygenRecord                = d_raygen_sbt.ptr,
            missRecordBase              = d_miss_sbt.ptr,
            missRecordStrideInBytes     = h_miss_sbt.dtype.itemsize,
            missRecordCount             = 1,
            hitgroupRecordBase          = d_hitgroup_sbt.ptr,
            hitgroupRecordStrideInBytes = h_hitgroup_sbt.dtype.itemsize,
            hitgroupRecordCount         = 1
        )
    
    def cast(self,rays):
        print( "Launching ... " )

        assert rays.shape[1] == 6, "Expected rays to be Nx6 [x,y,z,dx,dy,dz]"
        N = rays.shape[0]

        #origins = rays[:, 0:3].astype(np.float32).copy()
        #directions = rays[:, 3:6].astype(np.float32).copy()

        with self.stream:
            d_origins = rays[:, 0:3] #cp.asarray(origins)
            d_directions = rays[:, 3:6] #cp.asarray(directions)
            # d_hit_pos = cp.zeros((N, 3), dtype=cp.float32)
            # d_hit_dist = cp.zeros((N,), dtype=cp.float32)
            # d_hit_ids = cp.full((N,), 0xFFFFFFFF, dtype=cp.uint32)

            d_origins    = cp.ascontiguousarray(d_origins)
            d_directions = cp.ascontiguousarray(d_directions)

            d_hit_pos_inner = cp.ascontiguousarray(cp.zeros((N, 3), dtype=cp.float32))
            d_hit_dist_inner = cp.ascontiguousarray(cp.zeros((N,), dtype=cp.float32))
            d_hit_ids_inner = cp.ascontiguousarray(cp.full((N,), 0, dtype=cp.uint32))

            d_hit_pos_outer = cp.ascontiguousarray(cp.zeros((N, 3), dtype=cp.float32))
            d_hit_dist_outer = cp.ascontiguousarray(cp.zeros((N,), dtype=cp.float32))
            d_hit_ids_outer = cp.ascontiguousarray(cp.full((N,), 0, dtype=cp.uint32))

            d_hit_pos_tread = cp.ascontiguousarray(cp.zeros((N, 3), dtype=cp.float32))
            d_hit_dist_tread = cp.ascontiguousarray(cp.zeros((N,), dtype=cp.float32))
            d_hit_ids_tread = cp.ascontiguousarray(cp.full((N,), 0, dtype=cp.uint32))


        params = [
            ('u8', 'ray_origins', d_origins.data.ptr),
            ('u8', 'ray_directions', d_directions.data.ptr),
            ('u8', 'hit_positions_inner', d_hit_pos_inner.data.ptr),
            ('u8', 'hit_distances_inner', d_hit_dist_inner.data.ptr),
            ('u8', 'hit_tri_indices_inner', d_hit_ids_inner.data.ptr),
            ('u8', 'hit_positions_outer', d_hit_pos_outer.data.ptr),
            ('u8', 'hit_distances_outer', d_hit_dist_outer.data.ptr),
            ('u8', 'hit_tri_indices_outer', d_hit_ids_outer.data.ptr),
            ('u8', 'hit_positions_tread', d_hit_pos_tread.data.ptr),
            ('u8', 'hit_distances_tread', d_hit_dist_tread.data.ptr),
            ('u8', 'hit_tri_indices_tread', d_hit_ids_tread.data.ptr),
            ('u8', 'inner_handle', self.inner_gas_handle),
            ('u8', 'outer_handle', self.outer_gas_handle),
            ('u8', 'tread_handle', self.tread_gas_handle),
            ('u8', 'normals_inner', self.triangle_normals_inner.data.ptr),
            ('u4', 'num_rays', N),
        ]
        
        formats = [ x[0] for x in params ] 
        names   = [ x[1] for x in params ] 
        values  = [ x[2] for x in params ] 
        itemsize = get_aligned_itemsize( formats, 8 )
        params_dtype = np.dtype( { 
            'names'   : names, 
            'formats' : formats,
            'itemsize': itemsize,
            'align'   : True
            } )
        h_params = np.array( [ tuple(values) ], dtype=params_dtype )
        d_params = array_to_device_memory( h_params , self.stream)

        
        optix.launch( 
            self.pipeline, 
            self.stream.ptr, 
            d_params.ptr, 
            h_params.dtype.itemsize, 
            self.sbt,
            N,1,1
            )

        #self.stream.synchronize()
        return d_hit_pos_inner.copy(), d_hit_ids_inner.copy(), d_hit_dist_inner.copy(), d_hit_pos_outer.copy(), d_hit_ids_outer.copy(), d_hit_dist_outer.copy(), d_hit_pos_tread.copy(), d_hit_ids_tread.copy(), d_hit_dist_tread.copy()

    # def cast(self, rays):
    #     N = rays.shape[0]
    #     out_t = cp.zeros(N, dtype=cp.float32)
    #     out_id = cp.full(N, -1, dtype=cp.int32)
    #     out_xyz = cp.zeros((N, 3), dtype=cp.float32)

    #     d_rays = cp.ascontiguousarray(rays)

    #     params_dtype = np.dtype([
    #         ("rays", np.uint64),
    #         ("t_hit", np.uint64),
    #         ("tri_id", np.uint64),
    #         ("hit_point", np.uint64),
    #         ("num_rays", np.uint32)
    #     ])

    #     h_params = np.array([(int(d_rays.data.ptr), int(out_t.data.ptr), int(out_id.data.ptr), int(out_xyz.data.ptr), N)], dtype=params_dtype)
    #     d_params = cp.asarray(h_params)

    #     self.pipeline.launch(self.sbt, cp.cuda.Stream().ptr, int(d_params.data.ptr), h_params.itemsize, self.sbt, N, 1, 1)

    #     return out_t, out_id, out_xyz


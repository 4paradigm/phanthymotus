#!/usr/bin/env python3
"""Private TRT10.4 FP32/cuDNN plugin for the six pinned original encoder/acoustic LSTMs.

Module import is stdlib-only: no TensorRT import, CDLL, CUDA or NN.
Explicit make_plugin_classes(trt)/register_creator(trt) expose the old class API.
GPU resource creation happens only in explicit TensorRT workspace/shape callbacks.
The reviewed cuDNN 9.4 bindings live here alongside the TensorRT creator.
"""
import ctypes as ct
import hashlib
import json
import platform
import re
import threading
import time
import weakref
NAME='XiaoyouBiLSTMFP32';VERSION='1'
NAMESPACE='lanxijiang_xiaoyou_fp32_lstm_v2'
C=640;H=256;N_LIMIT=512;CAP_BYTES=512*2**20;WORKSPACE_BUDGET=64*2**20
ALLOWED_METADATA={'encoder:/prosody/lstms.0/LSTM': {'type': 'XiaoyouBiLSTMFP32', 'version': '1', 'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2', 'node_id': 'encoder:/prosody/lstms.0/LSTM', 'C': 640, 'H': 256, 'B': 1, 'N_limit': 512, 'precision': 'FP32', 'math': 'CUDNN_FMA_MATH', 'algorithm': 'STANDARD', 'input_layout': '[N,1,C]', 'output_layout': '[N,2,1,256]', 'gate_order': 'IOFC', 'h0_c0': 'zero_NULL', 'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL', 'original_weight_sha256': {'W': '4dc89597695a4a1854af96ce735ee3d7943d3de2abd5177802d1761e73120908', 'R': 'b0316424c11681f26d2e002bbe71f3b8f4832328c2e830b8de22c9a363702c61', 'B': 'f96fc22ad70d691f07b04cbe5b3c8627e21ace95478e17009970b0601ef2d839'}, 'weight_shapes': {'W': [2, 1024, 640], 'R': [2, 1024, 256], 'B': [2, 2048]}}, 'encoder:/prosody/lstms.2/LSTM': {'type': 'XiaoyouBiLSTMFP32', 'version': '1', 'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2', 'node_id': 'encoder:/prosody/lstms.2/LSTM', 'C': 640, 'H': 256, 'B': 1, 'N_limit': 512, 'precision': 'FP32', 'math': 'CUDNN_FMA_MATH', 'algorithm': 'STANDARD', 'input_layout': '[N,1,C]', 'output_layout': '[N,2,1,256]', 'gate_order': 'IOFC', 'h0_c0': 'zero_NULL', 'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL', 'original_weight_sha256': {'W': '8412d7352a4fb6aa50237c7f55d95ed5bbc3f8e9c445f0d98c0e52afdaa611b1', 'R': '4f259dd2e007fce89595d0d8ab2a17d441dc988b4c226bdc7b5adcd4937f228b', 'B': 'd17f91b45d66346857a779c3ae1c160c0f8f70acaea7349387810998313cd185'}, 'weight_shapes': {'W': [2, 1024, 640], 'R': [2, 1024, 256], 'B': [2, 2048]}}, 'encoder:/prosody/lstms.4/LSTM': {'type': 'XiaoyouBiLSTMFP32', 'version': '1', 'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2', 'node_id': 'encoder:/prosody/lstms.4/LSTM', 'C': 640, 'H': 256, 'B': 1, 'N_limit': 512, 'precision': 'FP32', 'math': 'CUDNN_FMA_MATH', 'algorithm': 'STANDARD', 'input_layout': '[N,1,C]', 'output_layout': '[N,2,1,256]', 'gate_order': 'IOFC', 'h0_c0': 'zero_NULL', 'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL', 'original_weight_sha256': {'W': 'f4f00ab7e7373470bb8f988c746797e282f5d87543d187cc1e062bb5f4f62800', 'R': 'ee4f983c070ab5253183e12eb02fa533d8b8a76b024991f044cc0fb9e84b17f0', 'B': '1f097376391686187054a739c37ab27001098f31929ad6e87184b70d9096ab1b'}, 'weight_shapes': {'W': [2, 1024, 640], 'R': [2, 1024, 256], 'B': [2, 2048]}}, 'encoder:/lstm/LSTM': {'type': 'XiaoyouBiLSTMFP32', 'version': '1', 'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2', 'node_id': 'encoder:/lstm/LSTM', 'C': 640, 'H': 256, 'B': 1, 'N_limit': 512, 'precision': 'FP32', 'math': 'CUDNN_FMA_MATH', 'algorithm': 'STANDARD', 'input_layout': '[N,1,C]', 'output_layout': '[N,2,1,256]', 'gate_order': 'IOFC', 'h0_c0': 'zero_NULL', 'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL', 'original_weight_sha256': {'W': 'ad196a98bc49edbf8847c4391be572653d96c2b69a84eefc5c73ac8315dbe859', 'R': '34332d1597ce57b348cf939d7ed60283d2538f71bddf38943ede7219745d3eee', 'B': '84fc4d8ac85f40e0841190eb3c8b2cde274b6b3192e8219931df0b420f8ff0c6'}, 'weight_shapes': {'W': [2, 1024, 640], 'R': [2, 1024, 256], 'B': [2, 2048]}}, 'encoder:/text_encoder/lstm/LSTM': {'type': 'XiaoyouBiLSTMFP32', 'version': '1', 'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2', 'node_id': 'encoder:/text_encoder/lstm/LSTM', 'C': 512, 'H': 256, 'B': 1, 'N_limit': 512, 'precision': 'FP32', 'math': 'CUDNN_FMA_MATH', 'algorithm': 'STANDARD', 'input_layout': '[N,1,C]', 'output_layout': '[N,2,1,256]', 'gate_order': 'IOFC', 'h0_c0': 'zero_NULL', 'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL', 'original_weight_sha256': {'W': '6fb6a58743cd381e03d813b04afc69a55b5267f35f3e6f3b4c86e654f99a795e', 'R': '023c76b13b03ab34c1a984b67c68fc7e872088826eb4d8af9afb7a40f6b7400d', 'B': 'ff46a9dca06149ec9c6b4147b824d66a66e5118f555961abe83795b32b9d8d54'}, 'weight_shapes': {'W': [2, 1024, 512], 'R': [2, 1024, 256], 'B': [2, 2048]}}, 'acoustic:/shared/LSTM': {'type': 'XiaoyouBiLSTMFP32', 'version': '1', 'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2', 'node_id': 'acoustic:/shared/LSTM', 'C': 640, 'H': 256, 'B': 1, 'N_limit': 832, 'precision': 'FP32', 'math': 'CUDNN_FMA_MATH', 'algorithm': 'STANDARD', 'input_layout': '[N,1,C]', 'output_layout': '[N,2,1,256]', 'gate_order': 'IOFC', 'h0_c0': 'zero_NULL', 'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL', 'original_weight_sha256': {'W': '61dbfe4e9b36df8518b89eb4fc6dcbb9a74f39d174a3cf6335436e22c39f0522', 'R': '38c8bc9a74b87ad34b76144eec3fe1f53deb1b42bbaf4fe84a4509d15da690ff', 'B': '60d30ff787954c278a6739d3f8eba483091ae25ce1962a186b9ffe34fd8f29a6'}, 'weight_shapes': {'W': [2, 1024, 640], 'R': [2, 1024, 256], 'B': [2, 2048]}}}
SOURCES=[
 'https://raw.githubusercontent.com/NVIDIA/TensorRT/release/10.3/python/src/infer/pyPlugin.cpp',
 'https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/plugins-python.html',
 'https://docs.nvidia.com/deeplearning/cudnn/backend/v9.4.0/api/cudnn-adv-library.html']
_CLASSES={};_REGISTERED={};_LOCK=threading.RLock();_INSTANCES=weakref.WeakValueDictionary()

P, I, S, U = ct.c_void_p, ct.c_int, ct.c_size_t, ct.c_uint32
PP, PI, PS = ct.POINTER(P), ct.POINTER(I), ct.POINTER(S)


def need(value, message):
    if not value:
        raise RuntimeError(message)


class APIError(RuntimeError):
    def __init__(self, library, operation, status, text):
        self.details = {"library": library, "operation": operation,
                        "status_code": int(status), "status_name": text}
        self.unsupported = library == "cuDNN" and "NOT_SUPPORTED" in text
        super().__init__(json.dumps(self.details))


class API:
    """Signatures are from the versioned NVIDIA docs and read-only headers above."""
    def __init__(self):
        need(ct.sizeof(P) == ct.sizeof(S) == 8 and ct.sizeof(I) == 4, "64-bit Linux ABI required")
        need(platform.system() == "Linux" and platform.machine() == "aarch64", "Reviewed JP6 target only")
        self.dnn = ct.CDLL("libcudnn.so.9")
        self.cuda = ct.CDLL("libcudart.so.12")
        self.bind(self.dnn, "cudnnGetVersion", S, [])
        self.bind(self.dnn, "cudnnGetErrorString", ct.c_char_p, [I])
        self.bind(self.cuda, "cudaGetErrorString", ct.c_char_p, [I])
        need(int(self.dnn.cudnnGetVersion()) == 90400, "Exact cuDNN 9.4.0 required; no version fallback")
        ds = {
            "cudnnCreate": [PP], "cudnnDestroy": [P], "cudnnSetStream": [P, P],
            "cudnnGetStream": [P, PP], "cudnnCreateRNNDescriptor": [PP],
            "cudnnDestroyRNNDescriptor": [P], "cudnnCreateRNNDataDescriptor": [PP],
            "cudnnDestroyRNNDataDescriptor": [P], "cudnnCreateTensorDescriptor": [PP],
            "cudnnDestroyTensorDescriptor": [P], "cudnnCreateDropoutDescriptor": [PP],
            "cudnnDestroyDropoutDescriptor": [P], "cudnnDropoutGetStatesSize": [P, PS],
            "cudnnSetDropoutDescriptor": [P, P, ct.c_float, P, S, ct.c_ulonglong],
            "cudnnSetRNNDescriptor_v8": [P] + [I] * 8 + [ct.c_int32] * 4 + [P, U],
            "cudnnGetRNNDescriptor_v8": [P] + [PI] * 12 + [PP, ct.POINTER(U)],
            "cudnnSetRNNDataDescriptor": [P, I, I, I, I, I, PI, P],
            "cudnnGetRNNTempSpaceSizes": [P, P, I, P, PS, PS],
            "cudnnGetRNNWeightSpaceSize": [P, P, PS],
            "cudnnGetRNNWeightParams": [P, P, ct.c_int32, S, P, ct.c_int32, P, PP, P, PP],
            "cudnnSetTensorNdDescriptor": [P, I, I, PI, PI],
            "cudnnGetTensorNdDescriptor": [P, I, PI, PI, PI, PI],
            "cudnnRNNForward": [P, P, I, P, P, P, P, P, P, P, P, P, P, P, S, P, S, P, S, P],
        }
        cs = {
            "cudaStreamCreate": [PP], "cudaStreamDestroy": [P], "cudaStreamSynchronize": [P],
            "cudaMalloc": [PP, S], "cudaFree": [P], "cudaMemcpyAsync": [P, P, S, I, P],
            "cudaMemsetAsync": [P, I, S, P], "cudaMemGetInfo": [PS, PS],
            "cudaRuntimeGetVersion": [PI], "cudaGetDevice": [PI],
            "cudaEventCreate": [PP], "cudaEventDestroy": [P],
            "cudaEventRecord": [P, P], "cudaEventSynchronize": [P],
            "cudaEventElapsedTime": [ct.POINTER(ct.c_float), P, P],
        }
        for name, args in ds.items():
            self.bind(self.dnn, name, I, args)
        for name, args in cs.items():
            self.bind(self.cuda, name, I, args)

    @staticmethod
    def bind(lib, name, result, args):
        function = getattr(lib, name)
        function.restype, function.argtypes = result, args

    def call(self, name, *args):
        is_dnn = name.startswith("cudnn")
        lib = self.dnn if is_dnn else self.cuda
        status = int(getattr(lib, name)(*args))
        if status:
            text = (self.dnn.cudnnGetErrorString(status) if is_dnn
                    else self.cuda.cudaGetErrorString(status))
            raise APIError("cuDNN" if is_dnn else "CUDA", name, status,
                           text.decode(errors="replace") if text else "unknown status")

    def memory(self):
        free, total = S(), S()
        self.call("cudaMemGetInfo", ct.byref(free), ct.byref(total))
        return {"free_bytes": free.value, "total_bytes": total.value}


class Resources:
    def __init__(self, api, report, stream):
        self.api, self.report = api, report
        self.descriptors, self.allocations, self.events = [], [], []
        self.total, self.peak = 0, 0
        self.stream, self.owns_stream, self.handle = P(), stream is None, P()
        if self.owns_stream:
            api.call("cudaStreamCreate", ct.byref(self.stream))
        else:
            need(isinstance(stream, int) and stream >= 0, "Invalid borrowed stream")
            self.stream = P(stream)
        try:
            api.call("cudnnCreate", ct.byref(self.handle))
            api.call("cudnnSetStream", self.handle, self.stream)
            actual = P()
            api.call("cudnnGetStream", self.handle, ct.byref(actual))
            need(actual.value == self.stream.value, "cuDNN stream readback mismatch")
        except BaseException:
            self.close()
            raise

    def descriptor(self, kind):
        value = P()
        self.api.call("cudnnCreate" + kind + "Descriptor", ct.byref(value))
        self.descriptors.append((kind, value))
        return value

    def allocate(self, name, size):
        size = int(size)
        need(0 <= size <= CAP_BYTES and self.total + size <= CAP_BYTES,
             "Explicit CUDA allocation cap 512MiB exceeded")
        if not size:
            return P()
        ptr = P()
        self.api.call("cudaMalloc", ct.byref(ptr), size)
        self.allocations.append((name, ptr, size))
        self.total += size
        self.peak = max(self.peak, self.total)
        return ptr

    def event(self):
        event = P()
        self.api.call("cudaEventCreate", ct.byref(event))
        self.events.append(event)
        return event

    def close(self):
        errors = []
        def attempt(name, *args):
            try:
                self.api.call(name, *args)
                return True
            except BaseException as error:
                errors.append(str(error))
                return False
        if self.stream or not self.owns_stream:
            attempt("cudaStreamSynchronize", self.stream)
        for event in reversed(self.events):
            attempt("cudaEventDestroy", event)
        for kind, value in reversed(self.descriptors):
            attempt("cudnnDestroy" + kind + "Descriptor", value)
        for _, ptr, size in reversed(self.allocations):
            if attempt("cudaFree", ptr):
                self.total -= size
        if self.handle:
            attempt("cudnnDestroy", self.handle)
        if self.owns_stream and self.stream:
            attempt("cudaStreamDestroy", self.stream)
        self.report["explicit_cuda_allocations"] = [
            {"name": name, "bytes": size} for name, _, size in self.allocations]
        self.report["explicit_cuda_peak_bytes"] = self.peak
        self.report["explicit_cuda_allocation_total_bytes"] = sum(size for _, _, size in self.allocations)
        self.report["explicit_cuda_unfreed_bytes"] = self.total
        self.report["cleanup_errors"] = errors
        self.report["cleanup_complete"] = not errors and self.total == 0
        self.events.clear()
        self.descriptors.clear()
        self.allocations.clear()
        self.handle = P()
        self.stream = P()


def tensor_shape(api, descriptor):
    dtype, ndim = I(), I()
    dims, strides = (I * 8)(), (I * 8)()
    api.call("cudnnGetTensorNdDescriptor", descriptor, 8, ct.byref(dtype),
             ct.byref(ndim), dims, strides)
    need(0 < ndim.value <= 8, "Unexpected tensor descriptor rank")
    return dtype.value, list(dims)[:ndim.value], list(strides)[:ndim.value]


def dense_strides(shape):
    result = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        result[i] = result[i + 1] * shape[i + 1]
    return result


def rnn_readback(api, descriptor):
    values, dropout, flags = [I() for _ in range(12)], P(), U()
    api.call("cudnnGetRNNDescriptor_v8", descriptor,
             *[ct.byref(v) for v in values], ct.byref(dropout), ct.byref(flags))
    names = ["algorithm", "cell_mode", "bias_mode", "direction", "input_mode",
             "data_type", "math_precision", "math_type", "input_size",
             "hidden_size", "projection_size", "num_layers"]
    return dict(zip(names, [v.value for v in values])), flags.value


def checked_metadata(value):
    need(isinstance(value,dict) and value.get('node_id') in ALLOWED_METADATA,'Unknown original LSTM')
    expected=ALLOWED_METADATA[value['node_id']]
    need(value==expected,'Original FP32 LSTM contract differs')
    return json.loads(json.dumps(expected))


class _Runtime:
    """One plugin instance's private handle/descriptors/weights; never shared by clone."""
    def __init__(self,weights,np,meta):
        self.C=meta['C'];self.N_limit=meta['N_limit']
        started=time.perf_counter()
        self.q=None;self.last_stream=None;self.current_N=None
        self.cache={};self.closed=False
        self.report={'prototype':True,'ready':False,'initialization_syncs':0,
            'enqueue_syncs':0,'enqueue_D2H':0,'enqueue_CPU_NN':False,
            'cap_scope':'plugin cudaMalloc + borrowed TRT workspace/input/output, excludes hidden library allocations',
            'workspace_budget_bytes':WORKSPACE_BUDGET,
            'workspace_bound_scope':'Fixed64MiB TRT scratch budget; query actual N and reject excess, no claim all N are proven'}
        self.api=API()
        try:
            self.q=q=Resources(self.api,self.report,None) # owns only this private initialization stream
            self.drop=q.descriptor('Dropout');self.rnn=q.descriptor('RNN')
            states=S();self.api.call('cudnnDropoutGetStatesSize',q.handle,ct.byref(states))
            self.drop_states=q.allocate('dropout_states',states.value)
            self.api.call('cudnnSetDropoutDescriptor',self.drop,q.handle,0.0,self.drop_states,states.value,42)
            self.api.call('cudnnSetRNNDescriptor_v8',self.rnn,0,2,2,1,0,0,0,3,self.C,H,H,1,self.drop,0)
            actual,flags=rnn_readback(self.api,self.rnn)
            need(list(actual.values())==[0,2,2,1,0,0,0,3,self.C,H,H,1]and flags==0,'Exact STANDARD FP32/FMA descriptor required')
            self.report['actual_RNN_descriptor']=actual
            self.xdesc=q.descriptor('RNNData');self.ydesc=q.descriptor('RNNData');self.query_desc=q.descriptor('RNNData')
            self.hdesc=q.descriptor('Tensor');self.cdesc=q.descriptor('Tensor')
            dims,strides=(I*3)(2,1,H),(I*3)(H,H,1)
            for descriptor in (self.hdesc,self.cdesc):self.api.call('cudnnSetTensorNdDescriptor',descriptor,0,3,dims,strides)
            weight_bytes=S();self.api.call('cudnnGetRNNWeightSpaceSize',q.handle,self.rnn,ct.byref(weight_bytes))
            self.weight_bytes=weight_bytes.value;self.weights=q.allocate('weights',self.weight_bytes)
            self.api.call('cudaMemsetAsync',self.weights,0,self.weight_bytes,q.stream)
            matrix_desc=q.descriptor('Tensor');bias_desc=q.descriptor('Tensor');intervals=[];keep=[]
            # Exactly the successful probe's IOFC -> IFCO matrix/bias mapping.
            for direction in range(2):
                for linear_id in range(8):
                    matrix_ptr,bias_ptr=P(),P()
                    self.api.call('cudnnGetRNNWeightParams',q.handle,self.rnn,direction,self.weight_bytes,self.weights,linear_id,matrix_desc,ct.byref(matrix_ptr),bias_desc,ct.byref(bias_ptr))
                    gate=(0,2,3,1)[linear_id%4];recurrent=linear_id>=4;width=H if recurrent else self.C;offset=4*H if recurrent else 0
                    matrix=np.ascontiguousarray(weights['R'if recurrent else'W'][direction,gate*H:(gate+1)*H,:])
                    bias=np.ascontiguousarray(weights['B'][direction,offset+gate*H:offset+(gate+1)*H])
                    for ptr,descriptor,array,wanted in [(matrix_ptr,matrix_desc,matrix,[1,H,width]),(bias_ptr,bias_desc,bias,[1,H,1])]:
                        dtype,shape,step=tensor_shape(self.api,descriptor)
                        need(dtype==0 and shape==wanted and step==dense_strides(shape),'Unexpected cuDNN parameter packing; no guessed transpose')
                        need(ptr.value is not None,'Missing original matrix/bias')
                        begin=ptr.value-self.weights.value;end=begin+array.nbytes
                        need(0<=begin<end<=self.weight_bytes and all(end<=a or begin>=b for a,b in intervals),'Invalid/overlapping weight space')
                        intervals.append((begin,end));keep.append(array)
                        self.api.call('cudaMemcpyAsync',ptr,P(array.ctypes.data),array.nbytes,1,q.stream)
            # Sole startup copy barrier, never inside enqueue. No device-to-host copy.
            self.api.call('cudaStreamSynchronize',q.stream);self.report['initialization_syncs']=1
            self.report['parameter_ranges']=len(intervals);self.report['weight_H2D_once']=True
            self.report['one_time_descriptor_and_weight_setup_wall_ms']=(time.perf_counter()-started)*1000
        except BaseException:
            self.close();raise

    def query(self,n):
        need(not self.closed and 1<=n<=self.N_limit,'N out of prototype bound')
        if n not in self.cache:
            seq=(I*1)(n);padding=ct.c_float(0)
            self.api.call('cudnnSetRNNDataDescriptor',self.query_desc,0,1,n,1,self.C,seq,ct.byref(padding))
            work,reserve=S(),S()
            self.api.call('cudnnGetRNNTempSpaceSizes',self.q.handle,self.rnn,0,self.query_desc,ct.byref(work),ct.byref(reserve))
            need(reserve.value==0,'Prototype supports only the observed zero inference reserve')
            need(work.value<=WORKSPACE_BUDGET,'Actual cuDNN workspace exceeds fixed64MiB prototype budget')
            need(self.q.total+WORKSPACE_BUDGET+n*(self.C+2*H)*4<=CAP_BYTES,'Explicit persistent+TRT scratch+I/O cap512MiB exceeded')
            self.cache[n]=work.value
            self.report.setdefault('actual_workspace_bytes_by_N',{})[str(n)]=work.value
        return self.cache[n]

    def shape(self,n):
        started=time.perf_counter()
        self.workspace_bytes=self.query(n);self.sequence=(I*1)(n);self.padding=ct.c_float(0)
        for descriptor,width in ((self.xdesc,self.C),(self.ydesc,2*H)):
            self.api.call('cudnnSetRNNDataDescriptor',descriptor,0,1,n,1,width,self.sequence,ct.byref(self.padding))
        self.current_N=n
        self.report.setdefault('shape_descriptor_query_wall_ms',[]).append({'N':n,'wall_ms':(time.perf_counter()-started)*1000})

    def enqueue(self,n,x,y,workspace,stream):
        need(not self.closed and self.current_N==n,'on_shape_change must finish before enqueue')
        need(isinstance(stream,int)and stream>=0 and x>0 and y>0,'Invalid borrowed pointers/stream')
        need(self.workspace_bytes==0 or workspace>0,'TensorRT workspace pointer required')
        self.last_stream=stream
        self.api.call('cudnnSetStream',self.q.handle,P(stream))
        # Exactly one whole-sequence NN call; no per-token Python loop/copy/sync.
        self.api.call('cudnnRNNForward',self.q.handle,self.rnn,0,None,self.xdesc,P(x),self.ydesc,P(y),self.hdesc,None,None,self.cdesc,None,None,self.weight_bytes,self.weights,self.workspace_bytes,P(workspace),0,None)

    def close(self):
        if self.closed:return
        self.closed=True
        if self.q is not None:
            # Teardown only. Borrowed TRT stream is synchronized, never destroyed.
            error=None
            try:
                if self.last_stream is not None:self.api.call('cudaStreamSynchronize',P(self.last_stream))
            except BaseException as caught:error=caught
            finally:self.q.close()
            need(self.report.get('cleanup_complete'),'Private cuDNN resource cleanup failed')
            if error is not None:raise error

def make_plugin_classes(trt):
    """Explicit class creation, still no CDLL/CUDA call. Pass the existing TRT module."""
    need(re.match(r'^10\.4(?:\.|$)',trt.__version__),'This prototype requires the tested TRT10.4 old class API')
    if id(trt)in _CLASSES:return _CLASSES[id(trt)]
    import numpy as np

    def own_weights(values,meta):
        shapes={k:tuple(v) for k,v in meta['weight_shapes'].items()};result={}
        need(set(values)==set(shapes),'Exactly original W/R/B are required')
        for name,shape in shapes.items():
            v=np.array(values[name],copy=True,order='C')
            need(v.dtype==np.float32 and v.size==int(np.prod(shape))and np.isfinite(v).all(),'Original finite FP32 weights only')
            v=v.reshape(shape);need(hashlib.sha256(v.tobytes()).hexdigest()==meta['original_weight_sha256'][name],'Exact original LSTM weights required')
            v.flags.writeable=False;result[name]=v
        return result

    def static_contract(inputs,outputs,meta):
        C=meta['C'];N_LIMIT=meta['N_limit']
        need(len(inputs)==len(outputs)==1,'One X and one Y only')
        x,y=inputs[0],outputs[0];n=int(x.dims[0])
        need(tuple(x.dims)==(n,1,C)and tuple(y.dims)==(n,2,1,H)and 1<=n<=N_LIMIT,'Exact B1/fullN ABI required')
        need(x.type==y.type==trt.DataType.FLOAT and x.format==y.format==trt.TensorFormat.LINEAR,'FP32 LINEAR only')
        return n

    class Plugin(trt.IPluginV3,trt.IPluginV3OneCore,trt.IPluginV3OneBuild,trt.IPluginV3OneRuntime):
        def __init__(self,weights,meta):
            for base in (trt.IPluginV3,trt.IPluginV3OneCore,trt.IPluginV3OneBuild,trt.IPluginV3OneRuntime):base.__init__(self)
            self.plugin_name=NAME;self.plugin_version=VERSION;self.plugin_namespace=NAMESPACE;self.num_outputs=1
            self.metadata=checked_metadata(meta);self.weights=own_weights(weights,self.metadata);self._state=None;self._mutex=threading.RLock();self._destroyed=False
            self._fields=None;self._field_storage=None;self.lifecycle_report={'prototype':True,'ready':False,'NN_validation':False}
            with _LOCK:_INSTANCES[id(self)]=self

        def get_capability_interface(self,kind):
            return self if kind in (trt.PluginCapabilityType.CORE,trt.PluginCapabilityType.BUILD,trt.PluginCapabilityType.RUNTIME)else None

        def get_valid_tactics(self):return []

        def set_tactic(self,tactic):need(tactic==0,'Only the default STANDARD/FMA tactic is supported')

        def clone(self):return Plugin(self.weights,self.metadata) # CPU-owned copies only; no device resource sharing/allocation

        def attach_to_context(self,resource_context):return self.clone()

        def get_output_data_types(self,input_types):
            need(len(input_types)==1 and input_types[0]==trt.DataType.FLOAT,'FP32 only');return [trt.DataType.FLOAT]

        def get_output_shapes(self,inputs,shape_inputs,expr_builder):
            need(len(inputs)==1 and len(inputs[0])==3 and not shape_inputs,'Only X[N,1,C]')
            return [trt.DimsExprs([inputs[0][0],expr_builder.constant(2),expr_builder.constant(1),expr_builder.constant(H)])]

        def supports_format_combination(self,pos,in_out,num_inputs):
            if num_inputs!=1 or len(in_out)!=2 or pos not in (0,1):return False
            desc=in_out[pos].desc
            return desc.type==trt.DataType.FLOAT and desc.format==trt.TensorFormat.LINEAR

        def configure_plugin(self,inputs,outputs):
            C=self.metadata['C'];N_LIMIT=self.metadata['N_limit']
            need(len(inputs)==len(outputs)==1,'Only one X/Y')
            for dynamic,width,rank in [(inputs[0],C,3),(outputs[0],H,4)]:
                desc=dynamic.desc;need(desc.type==trt.DataType.FLOAT and desc.format==trt.TensorFormat.LINEAR,'FP32 LINEAR only')
                need(len(desc.dims)==rank,'Wrong rank')
                for shape in (dynamic.min,dynamic.max):
                    n=int(shape[0]);need(1<=n<=N_LIMIT,'Bounded N only')
                    need(tuple(shape)==((n,1,C)if rank==3 else(n,2,1,H)),'Fixed B/C/H and output layout required')

        def _runtime(self):
            need(not self._destroyed,'Destroyed plugin')
            if self._state is None:self._state=_Runtime(self.weights,np,self.metadata)
            return self._state

        def get_workspace_size(self,inputs,outputs):
            self.configure_plugin(inputs,outputs);return WORKSPACE_BUDGET

        def on_shape_change(self,inputs,outputs):
            n=static_contract(inputs,outputs,self.metadata)
            with self._mutex:self._runtime().shape(n)

        def enqueue(self,input_desc,output_desc,inputs,outputs,workspace,stream):
            n=static_contract(input_desc,output_desc,self.metadata);need(len(inputs)==len(outputs)==1,'One input/output pointer')
            with self._mutex:
                need(self._state is not None,'Runtime must initialize before enqueue')
                self._state.enqueue(n,int(inputs[0]),int(outputs[0]),int(workspace),int(stream))

        def get_fields_to_serialize(self):
            # Keep own immutable backing arrays alive for the returned collection.
            self._field_storage={'metadata':np.frombuffer(json.dumps(self.metadata,sort_keys=True,separators=(',',':')).encode('utf-8'),dtype=np.int8).copy(),**{name:v.reshape(-1)for name,v in self.weights.items()}}
            self._fields=trt.PluginFieldCollection([trt.PluginField(name,value,trt.PluginFieldType.CHAR if name=='metadata'else trt.PluginFieldType.FLOAT32)for name,value in self._field_storage.items()])
            return self._fields

        def destroy(self):
            with self._mutex:
                if self._destroyed:return
                self._destroyed=True
                if self._state is not None:
                    try:self._state.close()
                    except BaseException as error:
                        self.lifecycle_report['explicit_teardown_error']=type(error).__name__+': '+str(error)
                        raise
                    finally:self.lifecycle_report.update(self._state.report);self._state=None

    class Creator(trt.IPluginCreatorV3One):
        def __init__(self):
            trt.IPluginCreatorV3One.__init__(self);self.name=NAME;self.plugin_version=VERSION;self.plugin_namespace=NAMESPACE
            self._empty={'metadata':np.empty(0,dtype=np.int8),**{name:np.empty(0,dtype=np.float32)for name in ('W','R','B')}}
            self.field_names=trt.PluginFieldCollection([trt.PluginField(name,value,trt.PluginFieldType.CHAR if name=='metadata'else trt.PluginFieldType.FLOAT32)for name,value in self._empty.items()])

        def create_plugin(self,name,fc,phase):
            need(phase in (trt.TensorRTPhase.BUILD,trt.TensorRTPhase.RUNTIME),'Unknown phase')
            fields={f.name:f for f in fc};need(len(fields)==len(fc)==4 and set(fields)=={'metadata','W','R','B'},'Exact serialized fields required')
            need(fields['metadata'].type==trt.PluginFieldType.CHAR,'Metadata CHAR required')
            payload=np.array(fields['metadata'].data,copy=True).tobytes();need(len(payload)<=4096,'Bounded metadata')
            # ONNX parser includes a C-string terminator; runtime serialization may omit it.
            payload=payload.rstrip(b'\x00')
            need(payload and b'\x00' not in payload,'Embedded metadata NUL forbidden')
            meta=checked_metadata(json.loads(payload.decode('utf-8')))
            weights={}
            for key in ('W','R','B'):
                need(fields[key].type==trt.PluginFieldType.FLOAT32,'FP32 weights only');weights[key]=np.array(fields[key].data,copy=True)
            return Plugin(weights,meta) # never retain TensorRT-owned field buffers

    _CLASSES[id(trt)]=(Plugin,Creator);return Plugin,Creator

def register_creator(trt):
    """Call explicitly before building/deserializing a plan in the SAME Python process."""
    with _LOCK:
        if id(trt)in _REGISTERED:return _REGISTERED[id(trt)]
        _,Creator=make_plugin_classes(trt);creator=Creator();registry=trt.get_plugin_registry()
        need(registry.get_creator(NAME,VERSION,NAMESPACE)is None,'Refuse collision with an existing creator')
        need(registry.register_creator(creator,NAMESPACE),'Own creator registration failed')
        _REGISTERED[id(trt)]=creator;return creator

def close_all_lifecycle():
    """Explicit unit-worker teardown BEFORE closing TRT pool/context or its stream.

    Caller must finish its original inference synchronization first. Only this
    module's plugin instances are weakly tracked; no strong global ownership and
    no other plugins, libraries, contexts or streams are destroyed here.
    The unit worker must require cleanup_complete before calling its result valid.
    """
    with _LOCK:instances=list(_INSTANCES.values())
    report={'scope':NAMESPACE,'instances':[],'errors':[],'cleanup_complete':True}
    for instance in instances:
        try:instance.destroy()
        except BaseException as error:report['errors'].append({'instance':id(instance),'error':type(error).__name__+': '+str(error)})
        report['instances'].append({'instance':id(instance),'destroyed':instance._destroyed,'state_released':instance._state is None,'lifecycle':dict(instance.lifecycle_report)})
    report['cleanup_complete']=not report['errors']and all(r['destroyed']and r['state_released']and r['lifecycle'].get('cleanup_complete',True)and not r['lifecycle'].get('explicit_teardown_error')for r in report['instances'])
    return report

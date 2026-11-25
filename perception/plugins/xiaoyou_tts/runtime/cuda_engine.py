"""TensorRT GPU execution only, with serialized shared activation memory.

No Torch/ORT/CPU neural implementation is imported. Named V3 and explicit-binding
V2 APIs are separated for version-specific engine packs; a plan is never rebuilt.
"""
import ctypes
import ctypes as ct
import hashlib
import json
import platform
import re
import time
import weakref
import ctypes.util
import gc
import math
import os
import threading

import numpy as np

from utils.tensorrt_runtime import trt_dtype_to_numpy


# JP6 FP32 LSTM creator and its reviewed cuDNN bindings; JP5 uses built-ins.
NAME='XiaoyouBiLSTMFP32';VERSION='1'
NAMESPACE='lanxijiang_xiaoyou_fp32_lstm_v2'
C=640;H=256;N_LIMIT=512;CAP_BYTES=512*2**20;WORKSPACE_BUDGET=64*2**20
ALLOWED_METADATA={'acoustic:/shared/LSTM': {'B': 1,
                           'C': 640,
                           'H': 256,
                           'N_limit': 832,
                           'algorithm': 'STANDARD',
                           'gate_order': 'IOFC',
                           'h0_c0': 'zero_NULL',
                           'input_layout': '[N,1,C]',
                           'math': 'CUDNN_FMA_MATH',
                           'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2',
                           'node_id': 'acoustic:/shared/LSTM',
                           'original_weight_sha256': {'B': '23e350e10ae487e1b45d4ffcd104ca52fed12153dcb7f0df4b50f5b5af32411e',
                                                      'R': '149fec453b09c58419d8ed4ddbc33ada87a1ad35d056a3f4f0dca3c0d984be52',
                                                      'W': 'ce3350825a08862b15f5b0933e0a4f0d294f952d67c45b1b835232ab9c6ced31'},
                           'output_layout': '[N,2,1,256]',
                           'precision': 'FP32',
                           'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL',
                           'type': 'XiaoyouBiLSTMFP32',
                           'version': '1',
                           'weight_shapes': {'B': [2, 2048], 'R': [2, 1024, 256], 'W': [2, 1024, 640]}},
 'encoder:/lstm/LSTM': {'B': 1,
                        'C': 640,
                        'H': 256,
                        'N_limit': 512,
                        'algorithm': 'STANDARD',
                        'gate_order': 'IOFC',
                        'h0_c0': 'zero_NULL',
                        'input_layout': '[N,1,C]',
                        'math': 'CUDNN_FMA_MATH',
                        'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2',
                        'node_id': 'encoder:/lstm/LSTM',
                        'original_weight_sha256': {'B': 'f1e2922384e88b4e1be68fd0e4c2cfe92e1c8ee1ef15def8fc2bdfbb5703e362',
                                                   'R': '8421c7ab1cb896fe82bae78e2b2bb65fa96b69c501e7f4a89ddafb7cbb56400c',
                                                   'W': '65300d103d91bfb401d1d9c2de10bc279758ed83bb95658f609949a2a868710e'},
                        'output_layout': '[N,2,1,256]',
                        'precision': 'FP32',
                        'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL',
                        'type': 'XiaoyouBiLSTMFP32',
                        'version': '1',
                        'weight_shapes': {'B': [2, 2048], 'R': [2, 1024, 256], 'W': [2, 1024, 640]}},
 'encoder:/prosody/lstms.0/LSTM': {'B': 1,
                                   'C': 640,
                                   'H': 256,
                                   'N_limit': 512,
                                   'algorithm': 'STANDARD',
                                   'gate_order': 'IOFC',
                                   'h0_c0': 'zero_NULL',
                                   'input_layout': '[N,1,C]',
                                   'math': 'CUDNN_FMA_MATH',
                                   'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2',
                                   'node_id': 'encoder:/prosody/lstms.0/LSTM',
                                   'original_weight_sha256': {'B': '55657ee87dfeefdc48509869cdce7ec7455740087f3c3fb60e891b1bc001944c',
                                                              'R': '7c0ef7dc0bbc438e865e25961dd508f1104195c921dd02abccc807bf2fafa4ba',
                                                              'W': '9bfd8791f28796f1fc0cee6617f79d42ab1eb2795aa3fdb619546f05635e647f'},
                                   'output_layout': '[N,2,1,256]',
                                   'precision': 'FP32',
                                   'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL',
                                   'type': 'XiaoyouBiLSTMFP32',
                                   'version': '1',
                                   'weight_shapes': {'B': [2, 2048],
                                                     'R': [2, 1024, 256],
                                                     'W': [2, 1024, 640]}},
 'encoder:/prosody/lstms.2/LSTM': {'B': 1,
                                   'C': 640,
                                   'H': 256,
                                   'N_limit': 512,
                                   'algorithm': 'STANDARD',
                                   'gate_order': 'IOFC',
                                   'h0_c0': 'zero_NULL',
                                   'input_layout': '[N,1,C]',
                                   'math': 'CUDNN_FMA_MATH',
                                   'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2',
                                   'node_id': 'encoder:/prosody/lstms.2/LSTM',
                                   'original_weight_sha256': {'B': 'bcbc9f3657f145f89460174021863b01d2db3d5c2dc5f44dbd20b1dc167b5227',
                                                              'R': '6059a3cb068dc83d2921085ad77358ae304fec54d0f76423ab249888c81d9da6',
                                                              'W': 'ac8de799881b262a4b982caef971e1cbf39dd1972e222cf1c5ab7f6958ecec3f'},
                                   'output_layout': '[N,2,1,256]',
                                   'precision': 'FP32',
                                   'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL',
                                   'type': 'XiaoyouBiLSTMFP32',
                                   'version': '1',
                                   'weight_shapes': {'B': [2, 2048],
                                                     'R': [2, 1024, 256],
                                                     'W': [2, 1024, 640]}},
 'encoder:/prosody/lstms.4/LSTM': {'B': 1,
                                   'C': 640,
                                   'H': 256,
                                   'N_limit': 512,
                                   'algorithm': 'STANDARD',
                                   'gate_order': 'IOFC',
                                   'h0_c0': 'zero_NULL',
                                   'input_layout': '[N,1,C]',
                                   'math': 'CUDNN_FMA_MATH',
                                   'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2',
                                   'node_id': 'encoder:/prosody/lstms.4/LSTM',
                                   'original_weight_sha256': {'B': '833983ab59aa8d1269da0571009504d246a16c45e6d3034905698371ab4a8182',
                                                              'R': 'ace5979356858fcd15fd82f0b2e8f48dd87a399c6fd5b8ec1da4d40c6e51c68f',
                                                              'W': 'f2b7c5d4f05d2b44b6a7468713a16da4d0d878f4c15cfcb30439f6384434c5c7'},
                                   'output_layout': '[N,2,1,256]',
                                   'precision': 'FP32',
                                   'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL',
                                   'type': 'XiaoyouBiLSTMFP32',
                                   'version': '1',
                                   'weight_shapes': {'B': [2, 2048],
                                                     'R': [2, 1024, 256],
                                                     'W': [2, 1024, 640]}},
 'encoder:/text_encoder/lstm/LSTM': {'B': 1,
                                     'C': 512,
                                     'H': 256,
                                     'N_limit': 512,
                                     'algorithm': 'STANDARD',
                                     'gate_order': 'IOFC',
                                     'h0_c0': 'zero_NULL',
                                     'input_layout': '[N,1,C]',
                                     'math': 'CUDNN_FMA_MATH',
                                     'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2',
                                     'node_id': 'encoder:/text_encoder/lstm/LSTM',
                                     'original_weight_sha256': {'B': 'ff46a9dca06149ec9c6b4147b824d66a66e5118f555961abe83795b32b9d8d54',
                                                                'R': '023c76b13b03ab34c1a984b67c68fc7e872088826eb4d8af9afb7a40f6b7400d',
                                                                'W': '6fb6a58743cd381e03d813b04afc69a55b5267f35f3e6f3b4c86e654f99a795e'},
                                     'output_layout': '[N,2,1,256]',
                                     'precision': 'FP32',
                                     'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL',
                                     'type': 'XiaoyouBiLSTMFP32',
                                     'version': '1',
                                     'weight_shapes': {'B': [2, 2048],
                                                       'R': [2, 1024, 256],
                                                       'W': [2, 1024, 512]}}}
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


class Cuda:
    def __init__(self):
        for name in (ctypes.util.find_library('cudart'), 'libcudart.so.12', 'libcudart.so.11.0', 'libcudart.so'):
            if not name:
                continue
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                pass
        else:
            raise RuntimeError('V6 requires a local CUDA runtime; no CPU fallback')
        signatures = {
            'cudaGetErrorString': ([ctypes.c_int], ctypes.c_char_p),
            'cudaSetDevice': ([ctypes.c_int], ctypes.c_int),
            'cudaMalloc': ([ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t], ctypes.c_int),
            'cudaFree': ([ctypes.c_void_p], ctypes.c_int),
            'cudaMemcpyAsync': ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
            'cudaStreamCreate': ([ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
            'cudaStreamSynchronize': ([ctypes.c_void_p], ctypes.c_int),
            'cudaStreamDestroy': ([ctypes.c_void_p], ctypes.c_int),
            'cudaMemGetInfo': ([ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)], ctypes.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(name + ' failed: ' + repr(self.lib.cudaGetErrorString(status)))

    def memory(self):
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.call('cudaMemGetInfo', ctypes.byref(free), ctypes.byref(total))
        return {'free_bytes': free.value, 'total_bytes': total.value}


class DeviceTensor:
    def __init__(self, pool, block, shape, dtype, *, _parent=None):
        self.pool, self.block = pool, block
        self._parent = _parent
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.nbytes = math.prod(self.shape) * self.dtype.itemsize
        self.released = False

    @property
    def pointer(self):
        if self.released or self.block['freed'] or self.pool.closed or self.pool.closing or self.pool.poisoned:
            raise RuntimeError('V6 device tensor is no longer live')
        if self._parent is not None:
            return self._parent.pointer
        return self.block['pointer']

    @property
    def readonly(self):
        return self._parent is not None

    def prefix_view(self, shape):
        """Borrow a dense read-only prefix; the parent retains its allocation."""
        with self.pool.lock:
            self.pointer  # Validate this owner before creating any borrowed view.
            shape = tuple(shape)
            if not shape or any(type(d) is not int or d <= 0 for d in shape) or math.prod(shape) > math.prod(self.shape):
                raise ValueError('V6 device prefix requires positive integer dimensions within its parent')
            return DeviceTensor(self.pool, self.block, shape, self.dtype, _parent=self)

    def numpy(self):
        return self.pool.download(self)

    def release(self):
        if not self.released:
            self.released = True
            if self._parent is None:
                self.pool.release(self.block)

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class _API:
    def __init__(self, engine, context, trt, *, legacy=False):
        self.engine, self.context, self.trt = engine, context, trt
        self.named = not legacy and hasattr(engine, 'num_io_tensors') and hasattr(context, 'execute_async_v3')
        if self.named:
            self.names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            self.inputs = {n for n in self.names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT}
        else:
            self.names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
            self.inputs = {n for n in self.names if engine.binding_is_input(engine.get_binding_index(n))}
        if len(set(self.names)) != len(self.names):
            raise RuntimeError('Duplicate TensorRT binding names')
        # infer_shapes() has no stream argument. Only defer host-copy waits when
        # engine metadata proves that input values cannot drive shape inference.
        self.can_defer_upload_sync = False
        query = getattr(engine, 'is_shape_inference_io' if self.named else 'is_shape_binding', None)
        if callable(query):
            try:
                self.can_defer_upload_sync = all(query(n if self.named else self.index(n)) is False for n in self.inputs)
            except Exception:
                self.can_defer_upload_sync = False

    def index(self, name):
        return self.engine.get_binding_index(name)

    def dtype(self, name):
        value = self.engine.get_tensor_dtype(name) if self.named else self.engine.get_binding_dtype(self.index(name))
        return trt_dtype_to_numpy(self.trt, value)

    def profile(self, name):
        value = self.engine.get_tensor_profile_shape(name, 0) if self.named else self.engine.get_profile_shape(0, name)
        return [list(v) for v in value]

    def require_device_linear(self, name):
        if self.named:
            if self.engine.get_tensor_location(name) != self.trt.TensorLocation.DEVICE:
                raise RuntimeError('V6 binding is not on GPU: ' + name)
            def get(method):
                fn = getattr(self.engine, method)
                try:
                    return fn(name, 0)
                except TypeError:
                    return fn(name)
            fmt = get('get_tensor_format')
            vector = int(get('get_tensor_vectorized_dim'))
            components = int(get('get_tensor_components_per_element'))
            width = int(get('get_tensor_bytes_per_component'))
        else:
            i = self.index(name)
            if self.engine.get_location(i) != self.trt.TensorLocation.DEVICE:
                raise RuntimeError('V6 legacy binding is not on GPU: ' + name)
            fmt = self.engine.get_binding_format(i)
            vector = int(self.engine.get_binding_vectorized_dim(i))
            # TRT 8.5 does not report component sizes consistently for linear outputs.
            # The device location, linear layout, vector dimension and dtype are checked.
            components = 1
            width = self.dtype(name).itemsize
        if fmt != self.trt.TensorFormat.LINEAR or vector != -1 or components not in (-1, 1) or width not in (-1, self.dtype(name).itemsize):
            raise RuntimeError('V6 requires scalar LINEAR GPU bindings: ' + name)

    def set_shape(self, name, shape):
        if self.named:
            self.context.set_input_shape(name, shape)
        else:
            self.context.set_binding_shape(self.index(name), shape)
        if self.shape(name) != tuple(shape):
            raise RuntimeError('V6 input shape readback differs: ' + name)

    def shape(self, name):
        value = self.context.get_tensor_shape(name) if self.named else self.context.get_binding_shape(self.index(name))
        return tuple(value)

    def require_shapes(self):
        if self.named:
            if self.context.infer_shapes():
                raise RuntimeError('V6 engine has unspecified shape tensors')
        elif not self.context.all_binding_shapes_specified:
            raise RuntimeError('V6 legacy binding shapes are incomplete')

    def strides(self, name):
        value = self.context.get_tensor_strides(name) if self.named else self.context.get_strides(self.index(name))
        return tuple(value)

    def addresses_and_execute(self, addresses, stream):
        if self.named:
            for name, pointer in addresses.items():
                self.context.set_tensor_address(name, pointer)
                if int(self.context.get_tensor_address(name)) != pointer:
                    raise RuntimeError('V6 tensor address readback mismatch')
            return self.context.execute_async_v3(stream_handle=stream)
        ordered = [addresses[n] for n in self.names]
        return self.context.execute_async_v2(bindings=ordered, stream_handle=stream)


class GPUEnginePool:
    """One GPU engine set and activation workspace, shared by logical sessions.

    Caller serializes jobs. This object additionally locks all allocation/execution
    paths so inference contexts and the shared workspace cannot execute concurrently.
    """
    def __init__(self, root, stages, *, expected_trt_prefix, max_explicit_bytes=1024 * 1024**2,
                 shape_workspace_cache_bytes=None):
        trt8 = expected_trt_prefix.startswith('8.')
        workspace_mode = os.environ.get('XIAOYOU01_WORKSPACE_MODE', 'fixed' if trt8 else 'shape')
        if workspace_mode not in ('fixed', 'shape'):
            raise ValueError('V6 workspace mode must be fixed or shape')
        if trt8 and workspace_mode != 'fixed':
            raise ValueError('JetPack 5 TensorRT requires fixed activation workspace')
        if shape_workspace_cache_bytes is None:
            default_cap = str(96 * 1024**2) if workspace_mode == 'shape' else '0'
            configured_cap = os.environ.get('XIAOYOU01_SHAPE_WORKSPACE_CACHE_BYTES', default_cap)
            if not configured_cap.isascii() or not configured_cap.isdecimal():
                raise ValueError('V6 shape workspace cache environment cap must be decimal bytes')
            shape_workspace_cache_bytes = int(configured_cap)
        if (type(shape_workspace_cache_bytes) is not int or shape_workspace_cache_bytes < 0
                or shape_workspace_cache_bytes > int(max_explicit_bytes)):
            raise ValueError('V6 shape workspace cache cap must be an integer within the explicit allocation cap')
        if shape_workspace_cache_bytes and workspace_mode != 'shape':
            raise ValueError('V6 shape workspace cache requires explicit shape workspace mode')
        nvtx_mode = os.environ.get('XIAOYOU01_NVTX_VERBOSITY', 'original')
        if nvtx_mode not in ('original', 'none'):
            raise ValueError('V6 NVTX verbosity must be original or none')
        import tensorrt as trt
        if not trt.__version__.startswith(expected_trt_prefix + '.') and trt.__version__ != expected_trt_prefix:
            raise RuntimeError('V6 engine pack requires TensorRT ' + expected_trt_prefix)
        self.trt, self.version = trt, trt.__version__
        self.cuda = Cuda()
        self.cuda.call('cudaSetDevice', 0)
        self.lock = threading.RLock()
        self.stream = ctypes.c_void_p()
        self.root, self.specs = root, stages
        self.closed = False
        self.closing = False
        self.poisoned = False
        self._stream_drained = True
        self._quarantined_refs = []
        self.blocks = {}
        self.cached_bytes = 0
        self.io_bytes = 0
        self._io_cache_primed = False
        self.max_explicit_bytes = int(max_explicit_bytes)
        self.max_cache_bytes = 64 * 1024**2
        self.workspace = None
        self.workspace_bytes = 0
        self.workspace_mode = workspace_mode
        # Opt-in only. At most one idle allocation per validated stage is kept,
        # with a strict total byte cap. These allocations remain in every budget.
        self.max_workspace_cache_bytes = shape_workspace_cache_bytes
        self._workspace_cache = {}
        self._workspace_cached_bytes = 0
        self._workspace_stage = None
        self.workspace_upper_bound_bytes = 0
        self.workspace_limits = {}
        self.engines = {}
        self._fused_lstm_registered = False
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = None
        engine = context = api = None
        try:
            if trt8:
                if not trt.init_libnvinfer_plugins(self.logger, ''):
                    raise RuntimeError('V6 TensorRT built-in plugin registration failed')
            else:
                register_creator(trt)
                self._fused_lstm_registered = True
            self.cuda.call('cudaStreamCreate', ctypes.byref(self.stream))
            self.runtime = trt.Runtime(self.logger)
            for name, spec in stages.items():
                engine = self.runtime.deserialize_cuda_engine((root / spec['file']).read_bytes())
                if engine is None or engine.num_optimization_profiles != 1:
                    raise RuntimeError('V6 engine/profile unavailable: ' + name)
                if hasattr(trt, 'ExecutionContextAllocationStrategy'):
                    context = engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
                elif self.workspace_mode == 'fixed':
                    context = engine.create_execution_context_without_device_memory()
                else:
                    raise RuntimeError('V6 shape workspace requires USER_MANAGED context allocation; no fallback')
                if context is None:
                    raise RuntimeError('V6 user-managed execution context unavailable')
                if nvtx_mode == 'none':
                    quiet = getattr(getattr(trt, 'ProfilingVerbosity', None), 'NONE', None)
                    if getattr(context, 'nvtx_verbosity', None) is None or quiet is None:
                        raise RuntimeError('V6 explicit NVTX none requires context/SDK support; no fallback')
                    context.nvtx_verbosity = quiet
                    if context.nvtx_verbosity != quiet:
                        raise RuntimeError('V6 NVTX verbosity readback differs; no fallback')
                api = _API(engine, context, trt, legacy=trt8)
                if set(api.names) != set(spec['inputs']) | set(spec['outputs']) or api.inputs != set(spec['inputs']):
                    raise RuntimeError('V6 engine I/O name/mode mismatch: ' + name)
                for tensor_name in api.names:
                    if api.dtype(tensor_name) != np.dtype(spec['io_dtypes'][tensor_name]):
                        raise RuntimeError('V6 engine I/O dtype mismatch: ' + tensor_name)
                    api.require_device_linear(tensor_name)
                    if tensor_name in api.inputs and api.profile(tensor_name) != spec['profiles'][tensor_name]:
                        raise RuntimeError('V6 engine profile mismatch: ' + tensor_name)
                upper = engine.device_memory_size
                if self.workspace_mode == 'shape':
                    if (not api.named or type(upper) is not int or upper <= 0
                            or not callable(getattr(context, 'update_device_memory_size_for_shapes', None))
                            or not callable(getattr(context, 'set_device_memory', None))):
                        raise RuntimeError('V6 shape workspace requires the verified TensorRT10.4 memory APIs')
                upper = int(upper)
                self.workspace_upper_bound_bytes = max(self.workspace_upper_bound_bytes, upper)
                self.workspace_limits[name] = upper
                self.engines[name] = api
                engine = context = api = None
            if self.workspace_mode == 'fixed':
                if self.workspace_upper_bound_bytes > self.max_explicit_bytes:
                    raise RuntimeError('V6 activation workspace exceeds explicit GPU allocation cap')
                if self.workspace_upper_bound_bytes:
                    self.workspace = ctypes.c_void_p()
                    self.cuda.call('cudaMalloc', ctypes.byref(self.workspace), self.workspace_upper_bound_bytes)
                    self.workspace_bytes = self.workspace_upper_bound_bytes
                    for obj in self.engines.values():
                        obj.context.device_memory = self.workspace.value
            self._prime_io_cache()
        except BaseException:
            engine = context = api = None
            self.close()
            raise

    def _prime_io_cache(self):
        """Reserve the verified three-stage binding capacities, with no NN/copy.

        Keep every lease live until all reservations exist: releasing each one
        immediately would repeatedly borrow the same best-fit block. One extra
        style slot stays available for the caller's subsequent persistent upload.
        Partial-stage diagnostic helpers retain their original allocation behavior.
        """
        signature = {
            'encoder': ({'tokens', 'style', 'speed'}, {'d', 't_en', 'duration'}),
            'acoustic': ({'d', 't_en', 'indices', 'style'}, {'features', 'f0'}),
            'vocoder': ({'features', 'f0', 'style', 'source_phase', 'excitation_noise'}, {'audio'}),
        }
        if set(self.specs) != set(signature) or any(
                (set(self.specs[k]['inputs']), set(self.specs[k]['outputs'])) != signature[k]
                for k in signature):
            return
        # The constructor calls this only after all original engine/profile/ABI
        # checks. Derive sizes from those actual metadata, never an audio probe.
        if set(self.engines) != set(signature) or self.blocks or self.io_bytes or self.cached_bytes:
            raise RuntimeError('V6 I/O cache priming requires the validated fresh three-stage pool')
        def maximum(stage, name):
            shape = tuple(self.specs[stage]['profiles'][name][-1])
            if not shape or any(type(d) is not int or d <= 0 for d in shape):
                raise RuntimeError('V6 I/O cache profile shape is invalid')
            return shape
        n_max = maximum('encoder', 'tokens')[-1]
        f_max = maximum('vocoder', 'f0')[-1]
        # The original certified operating range ends at F1460; acoustic F=2T.
        # Its physical acoustic profile is wider, so do not reserve T832 audio.
        if f_max != 1460 or f_max % 2:
            raise RuntimeError('V6 I/O cache requires the certified F1460 operating bound')
        source = maximum('vocoder', 'source_phase')
        if source != maximum('vocoder', 'excitation_noise'):
            raise RuntimeError('V6 I/O cache excitation profiles differ')
        symbols = {'N': n_max, 'T': min(maximum('acoustic', 'indices')[0], f_max // 2),
                   'F': f_max, 'samples': source[1]}
        reservations = [('encoder', 'style'), ('encoder', 'tokens'), ('encoder', 'speed'),
                        ('encoder', 'd'), ('encoder', 't_en'), ('encoder', 'duration'),
                        ('acoustic', 'indices'), ('acoustic', 'features'), ('acoustic', 'f0'),
                        ('vocoder', 'source_phase'), ('vocoder', 'excitation_noise'), ('vocoder', 'audio')]
        typed = []
        for stage, name in reservations:
            spec = self.specs[stage]
            if name == 'indices':
                shape = (symbols['T'],)
            elif name in spec['inputs']:
                shape = maximum(stage, name)
            else:
                shape = tuple(symbols[d] if isinstance(d, str) else d for d in spec['outputs'][name])
            dtype = np.dtype(spec['io_dtypes'][name])
            if not shape or any(type(d) is not int or d <= 0 for d in shape):
                raise RuntimeError('V6 I/O cache binding shape is invalid')
            typed.append((shape, dtype))
        total = sum(math.prod(shape) * dtype.itemsize for shape, dtype in typed)
        if total > self.max_cache_bytes:
            raise RuntimeError('V6 I/O cache priming exceeds the unchanged cache cap')
        if max(self.workspace_bytes, self.workspace_upper_bound_bytes) + total > self.max_explicit_bytes:
            raise RuntimeError('V6 I/O cache priming exceeds the unchanged explicit allocation cap')
        # On any failure the constructor's original close transaction owns all
        # allocated blocks, including leases still held by this traceback.
        leases = [self._allocate(shape, dtype) for shape, dtype in typed]
        if len({lease.pointer.value for lease in leases}) != len(typed):
            raise RuntimeError('V6 I/O cache priming did not reserve distinct live blocks')
        for lease in leases:
            lease.release()
        self._io_cache_primed = True

    def _require_live(self):
        if self.closed or self.closing:
            raise RuntimeError('V6 GPU engine pool is closed or closing')
        if self.poisoned:
            raise RuntimeError('V6 GPU engine pool is poisoned; no further inference')

    def _synchronize(self):
        try:
            self.cuda.call('cudaStreamSynchronize', self.stream)
        except BaseException:
            self.poisoned = True
            self._stream_drained = False
            raise
        self._stream_drained = True

    def _retain_until_drain(self, *references):
        # A failed sync cannot prove that DMA/kernel reads have stopped. Keep
        # actual host copies and tensor owners alive until a successful close.
        # Poison blocks new work, so this remains bounded to the failing call.
        self._quarantined_refs.append(references)

    def _activation_allocated_bytes(self):
        return self.workspace_bytes + self._workspace_cached_bytes

    def _evict_workspace(self, stage):
        block = self._workspace_cache.get(stage)
        if block is None:
            return
        # A previous lease was retired only after a drain. A new infer may have
        # queued H2D copies since then; explicitly drain before cudaFree anyway.
        if not self._stream_drained:
            self._synchronize()
        try:
            self.cuda.call('cudaFree', block['pointer'])
        except BaseException:
            self.poisoned = True
            raise
        self._workspace_cached_bytes -= block['bytes']
        del self._workspace_cache[stage]

    def _free_workspace_cache(self):
        for stage in list(self._workspace_cache):
            self._evict_workspace(stage)

    def _release_workspace(self, *, retain):
        """Retire an active lease after drain, or discard all on an error."""
        if self.workspace is not None:
            if not self._stream_drained:
                self.poisoned = True
                raise RuntimeError('V6 cannot retire activation memory before a confirmed stream drain')
            if (retain and self.max_workspace_cache_bytes
                    and self.workspace_bytes <= self.max_workspace_cache_bytes):
                stage = self._workspace_stage
                if stage is None or stage in self._workspace_cache:
                    self.poisoned = True
                    raise RuntimeError('V6 invalid shape workspace cache ownership')
                # Insertion order is LRU: renting removes an entry and returning
                # appends it. Never exceed the cap even momentarily while idle.
                while self._workspace_cached_bytes + self.workspace_bytes > self.max_workspace_cache_bytes:
                    self._evict_workspace(next(iter(self._workspace_cache)))
                self._workspace_cache[stage] = {'pointer': self.workspace, 'bytes': self.workspace_bytes}
                self._workspace_cached_bytes += self.workspace_bytes
                self.workspace = None
                self.workspace_bytes = 0
                self._workspace_stage = None
            else:
                self._free_workspace()
        if not retain:
            self._free_workspace_cache()

    def _rent_workspace(self, stage, api):
        if self.workspace is not None or self.workspace_bytes:
            raise RuntimeError('V6 shape workspace still has an outstanding lease')
        context = api.context
        profile = getattr(context, 'active_optimization_profile', None)
        if (not callable(getattr(context, 'update_device_memory_size_for_shapes', None))
                or not callable(getattr(context, 'set_device_memory', None))
                or type(profile) is not int or profile != 0):
            raise RuntimeError('V6 shape workspace API/profile unavailable; no fallback')
        required = context.update_device_memory_size_for_shapes()
        if type(required) is not int or not 0 < required <= self.workspace_limits[stage]:
            raise RuntimeError('V6 actual shape workspace size is invalid or exceeds the engine bound')
        block = self._workspace_cache.get(stage)
        # Reuse prefers this stage, then best-fit idle memory from any stage. A
        # temporary large shape cannot leave a large high-water block attached
        # to all later small requests: a >2x size drop reallocates at actual size.
        reuse = (block is not None and required <= block['bytes'] <= 2 * required)
        if block is not None and not reuse:
            self._evict_workspace(stage)
        source_stage = stage
        if not reuse:
            compatible = [(name, cached) for name, cached in self._workspace_cache.items()
                          if required <= cached['bytes'] <= 2 * required]
            if compatible:
                source_stage, block = min(compatible, key=lambda item: item[1]['bytes'])
                reuse = True
        if reuse:
            # Each idle block was retired after a confirmed stream drain. Remove
            # its old owner before leasing; return will label it for this stage.
            del self._workspace_cache[source_stage]
            self._workspace_cached_bytes -= block['bytes']
            pointer, capacity = block['pointer'], block['bytes']
        else:
            if self.io_bytes + self._activation_allocated_bytes() + required > self.max_explicit_bytes:
                self._trim_cache()
            while (self._workspace_cache
                   and self.io_bytes + self._activation_allocated_bytes() + required > self.max_explicit_bytes):
                self._evict_workspace(next(iter(self._workspace_cache)))
            if self.io_bytes + self._activation_allocated_bytes() + required > self.max_explicit_bytes:
                raise RuntimeError('V6 explicit GPU allocation cap exceeded by shape workspace; no fallback')
            pointer, capacity = ctypes.c_void_p(), required
            try:
                self.cuda.call('cudaMalloc', ctypes.byref(pointer), required)
            except BaseException:
                if pointer.value:
                    self.workspace, self.workspace_bytes = pointer, capacity
                    self._workspace_stage = stage
                    self.poisoned = True
                raise
            if not pointer.value:
                raise RuntimeError('V6 shape workspace allocation returned a null pointer')
        self.workspace, self.workspace_bytes = pointer, capacity
        self._workspace_stage = stage
        # TensorRT10.4's setter returns None on success. A context may retain
        # this address while idle; every shape-mode execute rebinds a fresh lease.
        context.set_device_memory(pointer.value, capacity)

    def _free_workspace(self):
        if self.workspace is None:
            return
        if not self._stream_drained:
            self.poisoned = True
            raise RuntimeError('V6 cannot free activation memory before a confirmed stream drain')
        try:
            self.cuda.call('cudaFree', self.workspace)
        except BaseException:
            self.poisoned = True
            raise
        self.workspace = None
        self.workspace_bytes = 0
        self._workspace_stage = None

    def _allocate(self, shape, dtype):
        self._require_live()
        size = math.prod(shape) * np.dtype(dtype).itemsize
        if size <= 0:
            raise ValueError('V6 cannot allocate an empty dynamic tensor')
        candidates = [b for b in self.blocks.values() if not b['in_use'] and not b['freed'] and b['bytes'] >= size]
        if candidates:
            block = min(candidates, key=lambda b: b['bytes'])
            self.cached_bytes -= block['bytes']
            block['in_use'] = True
        else:
            while (self._workspace_cache
                   and self._activation_allocated_bytes() + self.io_bytes + size > self.max_explicit_bytes):
                self._evict_workspace(next(iter(self._workspace_cache)))
            if self._activation_allocated_bytes() + self.io_bytes + size > self.max_explicit_bytes:
                self._trim_cache()
            if self._activation_allocated_bytes() + self.io_bytes + size > self.max_explicit_bytes:
                raise RuntimeError('V6 explicit GPU allocation cap exceeded; no fallback')
            pointer = ctypes.c_void_p()
            self.cuda.call('cudaMalloc', ctypes.byref(pointer), size)
            block = {'pointer': pointer, 'bytes': size, 'in_use': True, 'freed': False}
            self.blocks[pointer.value] = block
            self.io_bytes += size
        return DeviceTensor(self, block, shape, dtype)

    def _trim_cache(self):
        for block in list(self.blocks.values()):
            if not block['in_use'] and not block['freed']:
                self.cuda.call('cudaFree', block['pointer'])
                block['freed'] = True
                self.io_bytes -= block['bytes']
                self.cached_bytes -= block['bytes']
                del self.blocks[block['pointer'].value]

    def release(self, block):
        with self.lock:
            if block['freed'] or not block['in_use']:
                return
            if not self._stream_drained:
                block['release_deferred'] = True
                return
            self.cuda.call('cudaSetDevice', 0)
            block['in_use'] = False
            self.cached_bytes += block['bytes']
            if self.cached_bytes > self.max_cache_bytes:
                self._trim_cache()

    def upload(self, array):
        with self.lock:
            self._require_live()
            self.cuda.call('cudaSetDevice', 0)
            a = out = None
            try:
                a = np.ascontiguousarray(array)
                if not np.isfinite(a).all():
                    raise ValueError('V6 upload contains nonfinite data')
                out = self._allocate(a.shape, a.dtype)
                self._stream_drained = False
                self.cuda.call('cudaMemcpyAsync', out.pointer, ctypes.c_void_p(a.ctypes.data), a.nbytes, 1, self.stream)
                self._synchronize()
                return out
            except BaseException:
                try:
                    self._synchronize()
                finally:
                    if self._stream_drained:
                        if out is not None:
                            out.release()
                        self._free_workspace_cache()
                    else:
                        self._retain_until_drain(a, out)
                raise

    def _enqueue_upload(self, array, temporary, host_sources):
        """Queue an infer-owned copy; caller retains buffers through stream sync."""
        self._require_live()
        self.cuda.call('cudaSetDevice', 0)
        a = np.ascontiguousarray(array)
        if not np.isfinite(a).all():
            raise ValueError('V6 upload contains nonfinite data')
        out = self._allocate(a.shape, a.dtype)
        # Register both owners before enqueue: CUDA can still read a temporary
        # contiguous source after this method returns or if a later input fails.
        temporary.append(out)
        host_sources.append(a)
        self._stream_drained = False
        self.cuda.call('cudaMemcpyAsync', out.pointer, ctypes.c_void_p(a.ctypes.data), a.nbytes, 1, self.stream)
        return out

    def download(self, tensor):
        with self.lock:
            self._require_live()
            if tensor.pool is not self:
                self._free_workspace_cache()
                raise RuntimeError('V6 device tensor belongs to a closed/different pool')
            self.cuda.call('cudaSetDevice', 0)
            result = np.empty(tensor.shape, dtype=tensor.dtype)
            try:
                self._stream_drained = False
                self.cuda.call('cudaMemcpyAsync', ctypes.c_void_p(result.ctypes.data), tensor.pointer, result.nbytes, 2, self.stream)
                self._synchronize()
            except BaseException:
                try:
                    self._synchronize()
                finally:
                    if not self._stream_drained:
                        self._retain_until_drain(tensor, result)
                    else:
                        self._free_workspace_cache()
                raise
            if not np.isfinite(result).all():
                self._free_workspace_cache()
                raise RuntimeError('V6 GPU output is nonfinite')
            return result

    def infer(self, stage, feed, expected_outputs):
        with self.lock:
            self._require_live()
            self.cuda.call('cudaSetDevice', 0)
            inputs, temporary, outputs = {}, [], {}
            host_sources = []
            completed = False
            try:
                spec, api = self.specs[stage], self.engines[stage]
                if set(feed) != set(spec['inputs']) or set(expected_outputs) != set(spec['outputs']):
                    raise ValueError('V6 stage routing differs: ' + stage)
                for name, value in feed.items():
                    if not isinstance(value, DeviceTensor):
                        if api.can_defer_upload_sync:
                            value = self._enqueue_upload(value, temporary, host_sources)
                        else:
                            value = self.upload(value)
                            temporary.append(value)
                    if value.pool is not self:
                        raise ValueError('V6 cannot mix device tensors from different engine pools')
                    if value.dtype != np.dtype(spec['io_dtypes'][name]):
                        raise ValueError('V6 input dtype differs: ' + name)
                    profile = spec['profiles'][name]
                    if len(value.shape) != len(profile[0]) or any(not lo <= x <= hi for x, lo, hi in zip(value.shape, profile[0], profile[2])):
                        raise ValueError('V6 input exceeds the verified GPU profile: ' + name)
                    api.set_shape(name, value.shape)
                    inputs[name] = value
                api.require_shapes()
                for name in api.names:
                    shape = api.shape(name)
                    expected = inputs[name].shape if name in inputs else tuple(expected_outputs[name])
                    if shape != expected or any(int(d) <= 0 for d in shape):
                        raise RuntimeError('V6 actual GPU output shape differs: ' + name)
                    stride = api.strides(name)
                    dense, product = [], 1
                    for d in reversed(shape):
                        dense.append(product)
                        product *= d
                    dense = tuple(reversed(dense))
                    if len(stride) != len(shape) or any(d != 1 and a != b for d, a, b in zip(shape, stride, dense)):
                        raise RuntimeError('V6 GPU binding is not physically dense: ' + name)
                    if name not in inputs:
                        outputs[name] = self._allocate(shape, spec['io_dtypes'][name])
                if self.workspace_mode == 'shape':
                    self._rent_workspace(stage, api)
                addresses = {name: tensor.pointer.value for name, tensor in {**inputs, **outputs}.items()}
                self._stream_drained = False
                if not api.addresses_and_execute(addresses, self.stream.value):
                    raise RuntimeError('V6 TensorRT execution failed: ' + stage)
                self._synchronize()
                completed = True
                return outputs
            except BaseException:
                try:
                    self._synchronize()
                finally:
                    if self._stream_drained:
                        for value in outputs.values():
                            value.release()
                    else:
                        self._retain_until_drain(inputs, temporary, outputs, host_sources)
                raise
            finally:
                if self._stream_drained:
                    try:
                        if self.workspace_mode == 'shape':
                            self._release_workspace(retain=completed)
                    finally:
                        for value in temporary:
                            value.release()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closing = True
            try:
                self.cuda.call('cudaSetDevice', 0)
                if self.stream.value:
                    self._synchronize()
                # Destruction/free only follows a confirmed drain. On failure,
                # keep owners and the live stream so close() can be retried;
                # the pool stays poisoned and rejects all subsequent inference.
                if self._fused_lstm_registered:
                    fused_cleanup = close_all_lifecycle()
                    if not fused_cleanup['cleanup_complete']:
                        raise RuntimeError('Private fused LSTM resource cleanup failed')
                for obj in self.engines.values():
                    obj.context = None
                    obj.engine = None
                self.engines.clear()
                gc.collect()
                for block in self.blocks.values():
                    if not block['freed']:
                        self.cuda.call('cudaFree', block['pointer'])
                        block['freed'] = True
                        self.io_bytes -= block['bytes']
                        if not block['in_use']:
                            self.cached_bytes -= block['bytes']
                self.blocks.clear()
                self._free_workspace()
                self._free_workspace_cache()
                self.runtime = None
                gc.collect()
                if self.stream.value:
                    self.cuda.call('cudaStreamDestroy', self.stream)
                    self.stream = ctypes.c_void_p()
                self._quarantined_refs.clear()
            except BaseException:
                self.poisoned = True
                raise
            self.closed = True
            self.closing = False

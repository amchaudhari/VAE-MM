??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
decoder/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
?*'
shared_namedecoder/dense_6/kernel
?
*decoder/dense_6/kernel/Read/ReadVariableOpReadVariableOpdecoder/dense_6/kernel*
_output_shapes
:	
?*
dtype0
?
decoder/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_namedecoder/dense_6/bias
z
(decoder/dense_6/bias/Read/ReadVariableOpReadVariableOpdecoder/dense_6/bias*
_output_shapes	
:?*
dtype0
?
!decoder/conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!decoder/conv2d_transpose_3/kernel
?
5decoder/conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_3/kernel*&
_output_shapes
:@@*
dtype0
?
decoder/conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!decoder/conv2d_transpose_3/bias
?
3decoder/conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_3/bias*
_output_shapes
:@*
dtype0
?
!decoder/conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!decoder/conv2d_transpose_4/kernel
?
5decoder/conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_4/kernel*&
_output_shapes
: @*
dtype0
?
decoder/conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!decoder/conv2d_transpose_4/bias
?
3decoder/conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_4/bias*
_output_shapes
: *
dtype0
?
!decoder/conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!decoder/conv2d_transpose_5/kernel
?
5decoder/conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_5/kernel*&
_output_shapes
: *
dtype0
?
decoder/conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!decoder/conv2d_transpose_5/bias
?
3decoder/conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_5/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
	dense
reshape
	conv1
	conv2
	conv3
regularization_losses
	variables
trainable_variables
		keras_api


signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
h

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
 
8
0
1
2
3
4
5
!6
"7
8
0
1
2
3
4
5
!6
"7
?
'layer_metrics

(layers
)layer_regularization_losses
*metrics
regularization_losses
	variables
+non_trainable_variables
trainable_variables
 
SQ
VARIABLE_VALUEdecoder/dense_6/kernel'dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdecoder/dense_6/bias%dense/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
,layer_metrics
-layer_regularization_losses
.metrics
regularization_losses

/layers
0non_trainable_variables
trainable_variables
 
 
 
?
	variables
1layer_metrics
2layer_regularization_losses
3metrics
regularization_losses

4layers
5non_trainable_variables
trainable_variables
^\
VARIABLE_VALUE!decoder/conv2d_transpose_3/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdecoder/conv2d_transpose_3/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
6layer_metrics
7layer_regularization_losses
8metrics
regularization_losses

9layers
:non_trainable_variables
trainable_variables
^\
VARIABLE_VALUE!decoder/conv2d_transpose_4/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdecoder/conv2d_transpose_4/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
;layer_metrics
<layer_regularization_losses
=metrics
regularization_losses

>layers
?non_trainable_variables
trainable_variables
^\
VARIABLE_VALUE!decoder/conv2d_transpose_5/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdecoder/conv2d_transpose_5/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
?
#	variables
@layer_metrics
Alayer_regularization_losses
Bmetrics
$regularization_losses

Clayers
Dnon_trainable_variables
%trainable_variables
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????
*
dtype0*
shape:?????????

?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1decoder/dense_6/kerneldecoder/dense_6/bias!decoder/conv2d_transpose_3/kerneldecoder/conv2d_transpose_3/bias!decoder/conv2d_transpose_4/kerneldecoder/conv2d_transpose_4/bias!decoder/conv2d_transpose_5/kerneldecoder/conv2d_transpose_5/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_121509
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*decoder/dense_6/kernel/Read/ReadVariableOp(decoder/dense_6/bias/Read/ReadVariableOp5decoder/conv2d_transpose_3/kernel/Read/ReadVariableOp3decoder/conv2d_transpose_3/bias/Read/ReadVariableOp5decoder/conv2d_transpose_4/kernel/Read/ReadVariableOp3decoder/conv2d_transpose_4/bias/Read/ReadVariableOp5decoder/conv2d_transpose_5/kernel/Read/ReadVariableOp3decoder/conv2d_transpose_5/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_121595
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedecoder/dense_6/kerneldecoder/dense_6/bias!decoder/conv2d_transpose_3/kerneldecoder/conv2d_transpose_3/bias!decoder/conv2d_transpose_4/kerneldecoder/conv2d_transpose_4/bias!decoder/conv2d_transpose_5/kerneldecoder/conv2d_transpose_5/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_121629??
?	
?
C__inference_dense_6_layer_call_and_return_conditional_losses_121410

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_121543

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_121509
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_1212602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????
::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
?$
?
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_121385

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?$
?
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_121295

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
ƈ
?
!__inference__wrapped_model_121260
input_12
.decoder_dense_6_matmul_readvariableop_resource3
/decoder_dense_6_biasadd_readvariableop_resourceG
Cdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource>
:decoder_conv2d_transpose_3_biasadd_readvariableop_resourceG
Cdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource>
:decoder_conv2d_transpose_4_biasadd_readvariableop_resourceG
Cdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource>
:decoder_conv2d_transpose_5_biasadd_readvariableop_resource
identity??1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?&decoder/dense_6/BiasAdd/ReadVariableOp?%decoder/dense_6/MatMul/ReadVariableOp?
%decoder/dense_6/MatMul/ReadVariableOpReadVariableOp.decoder_dense_6_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype02'
%decoder/dense_6/MatMul/ReadVariableOp?
decoder/dense_6/MatMulMatMulinput_1-decoder/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_6/MatMul?
&decoder/dense_6/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&decoder/dense_6/BiasAdd/ReadVariableOp?
decoder/dense_6/BiasAddBiasAdd decoder/dense_6/MatMul:product:0.decoder/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_6/BiasAdd?
decoder/dense_6/ReluRelu decoder/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
decoder/dense_6/Relu?
decoder/reshape_1/ShapeShape"decoder/dense_6/Relu:activations:0*
T0*
_output_shapes
:2
decoder/reshape_1/Shape?
%decoder/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%decoder/reshape_1/strided_slice/stack?
'decoder/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder/reshape_1/strided_slice/stack_1?
'decoder/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder/reshape_1/strided_slice/stack_2?
decoder/reshape_1/strided_sliceStridedSlice decoder/reshape_1/Shape:output:0.decoder/reshape_1/strided_slice/stack:output:00decoder/reshape_1/strided_slice/stack_1:output:00decoder/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
decoder/reshape_1/strided_slice?
!decoder/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/reshape_1/Reshape/shape/1?
!decoder/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/reshape_1/Reshape/shape/2?
!decoder/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2#
!decoder/reshape_1/Reshape/shape/3?
decoder/reshape_1/Reshape/shapePack(decoder/reshape_1/strided_slice:output:0*decoder/reshape_1/Reshape/shape/1:output:0*decoder/reshape_1/Reshape/shape/2:output:0*decoder/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
decoder/reshape_1/Reshape/shape?
decoder/reshape_1/ReshapeReshape"decoder/dense_6/Relu:activations:0(decoder/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@2
decoder/reshape_1/Reshape?
 decoder/conv2d_transpose_3/ShapeShape"decoder/reshape_1/Reshape:output:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_3/Shape?
.decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_3/strided_slice/stack?
0decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_3/strided_slice/stack_1?
0decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_3/strided_slice/stack_2?
(decoder/conv2d_transpose_3/strided_sliceStridedSlice)decoder/conv2d_transpose_3/Shape:output:07decoder/conv2d_transpose_3/strided_slice/stack:output:09decoder/conv2d_transpose_3/strided_slice/stack_1:output:09decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_3/strided_slice?
"decoder/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_3/stack/1?
"decoder/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_3/stack/2?
"decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2$
"decoder/conv2d_transpose_3/stack/3?
 decoder/conv2d_transpose_3/stackPack1decoder/conv2d_transpose_3/strided_slice:output:0+decoder/conv2d_transpose_3/stack/1:output:0+decoder/conv2d_transpose_3/stack/2:output:0+decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_3/stack?
0decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_3/strided_slice_1/stack?
2decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_1/stack_1?
2decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_1/stack_2?
*decoder/conv2d_transpose_3/strided_slice_1StridedSlice)decoder/conv2d_transpose_3/stack:output:09decoder/conv2d_transpose_3/strided_slice_1/stack:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_3/strided_slice_1?
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02<
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?
+decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_3/stack:output:0Bdecoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0"decoder/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2-
+decoder/conv2d_transpose_3/conv2d_transpose?
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp?
"decoder/conv2d_transpose_3/BiasAddBiasAdd4decoder/conv2d_transpose_3/conv2d_transpose:output:09decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2$
"decoder/conv2d_transpose_3/BiasAdd?
decoder/conv2d_transpose_3/ReluRelu+decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2!
decoder/conv2d_transpose_3/Relu?
 decoder/conv2d_transpose_4/ShapeShape-decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_4/Shape?
.decoder/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_4/strided_slice/stack?
0decoder/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_4/strided_slice/stack_1?
0decoder/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_4/strided_slice/stack_2?
(decoder/conv2d_transpose_4/strided_sliceStridedSlice)decoder/conv2d_transpose_4/Shape:output:07decoder/conv2d_transpose_4/strided_slice/stack:output:09decoder/conv2d_transpose_4/strided_slice/stack_1:output:09decoder/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_4/strided_slice?
"decoder/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_4/stack/1?
"decoder/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_4/stack/2?
"decoder/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2$
"decoder/conv2d_transpose_4/stack/3?
 decoder/conv2d_transpose_4/stackPack1decoder/conv2d_transpose_4/strided_slice:output:0+decoder/conv2d_transpose_4/stack/1:output:0+decoder/conv2d_transpose_4/stack/2:output:0+decoder/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_4/stack?
0decoder/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_4/strided_slice_1/stack?
2decoder/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_1/stack_1?
2decoder/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_1/stack_2?
*decoder/conv2d_transpose_4/strided_slice_1StridedSlice)decoder/conv2d_transpose_4/stack:output:09decoder/conv2d_transpose_4/strided_slice_1/stack:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_4/strided_slice_1?
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02<
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?
+decoder/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_4/stack:output:0Bdecoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2-
+decoder/conv2d_transpose_4/conv2d_transpose?
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp?
"decoder/conv2d_transpose_4/BiasAddBiasAdd4decoder/conv2d_transpose_4/conv2d_transpose:output:09decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2$
"decoder/conv2d_transpose_4/BiasAdd?
decoder/conv2d_transpose_4/ReluRelu+decoder/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2!
decoder/conv2d_transpose_4/Relu?
 decoder/conv2d_transpose_5/ShapeShape-decoder/conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_5/Shape?
.decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_5/strided_slice/stack?
0decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_5/strided_slice/stack_1?
0decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_5/strided_slice/stack_2?
(decoder/conv2d_transpose_5/strided_sliceStridedSlice)decoder/conv2d_transpose_5/Shape:output:07decoder/conv2d_transpose_5/strided_slice/stack:output:09decoder/conv2d_transpose_5/strided_slice/stack_1:output:09decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_5/strided_slice?
"decoder/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_5/stack/1?
"decoder/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_5/stack/2?
"decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_5/stack/3?
 decoder/conv2d_transpose_5/stackPack1decoder/conv2d_transpose_5/strided_slice:output:0+decoder/conv2d_transpose_5/stack/1:output:0+decoder/conv2d_transpose_5/stack/2:output:0+decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_5/stack?
0decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_5/strided_slice_1/stack?
2decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_1/stack_1?
2decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_1/stack_2?
*decoder/conv2d_transpose_5/strided_slice_1StridedSlice)decoder/conv2d_transpose_5/stack:output:09decoder/conv2d_transpose_5/strided_slice_1/stack:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_5/strided_slice_1?
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02<
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?
+decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_5/stack:output:0Bdecoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2-
+decoder/conv2d_transpose_5/conv2d_transpose?
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp?
"decoder/conv2d_transpose_5/BiasAddBiasAdd4decoder/conv2d_transpose_5/conv2d_transpose:output:09decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2$
"decoder/conv2d_transpose_5/BiasAdd?
"decoder/conv2d_transpose_5/SigmoidSigmoid+decoder/conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2$
"decoder/conv2d_transpose_5/Sigmoid?
IdentityIdentity&decoder/conv2d_transpose_5/Sigmoid:y:02^decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp'^decoder/dense_6/BiasAdd/ReadVariableOp&^decoder/dense_6/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????
::::::::2f
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2P
&decoder/dense_6/BiasAdd/ReadVariableOp&decoder/dense_6/BiasAdd/ReadVariableOp2N
%decoder/dense_6/MatMul/ReadVariableOp%decoder/dense_6/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
?
}
(__inference_dense_6_layer_call_fn_121529

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1214102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
(__inference_decoder_layer_call_fn_121486
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_1214642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????
::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
?
?
C__inference_decoder_layer_call_and_return_conditional_losses_121464
input_1
dense_6_121421
dense_6_121423
conv2d_transpose_3_121448
conv2d_transpose_3_121450
conv2d_transpose_4_121453
conv2d_transpose_4_121455
conv2d_transpose_5_121458
conv2d_transpose_5_121460
identity??*conv2d_transpose_3/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_6_121421dense_6_121423*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1214102!
dense_6/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1214402
reshape_1/PartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_3_121448conv2d_transpose_3_121450*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1212952,
*conv2d_transpose_3/StatefulPartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_121453conv2d_transpose_4_121455*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1213402,
*conv2d_transpose_4/StatefulPartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_121458conv2d_transpose_5_121460*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_1213852,
*conv2d_transpose_5/StatefulPartitionedCall?
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????
::::::::2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
?
F
*__inference_reshape_1_layer_call_fn_121548

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1214402
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?&
?
"__inference__traced_restore_121629
file_prefix+
'assignvariableop_decoder_dense_6_kernel+
'assignvariableop_1_decoder_dense_6_bias8
4assignvariableop_2_decoder_conv2d_transpose_3_kernel6
2assignvariableop_3_decoder_conv2d_transpose_3_bias8
4assignvariableop_4_decoder_conv2d_transpose_4_kernel6
2assignvariableop_5_decoder_conv2d_transpose_4_bias8
4assignvariableop_6_decoder_conv2d_transpose_5_kernel6
2assignvariableop_7_decoder_conv2d_transpose_5_bias

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp'assignvariableop_decoder_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp'assignvariableop_1_decoder_dense_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp4assignvariableop_2_decoder_conv2d_transpose_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp2assignvariableop_3_decoder_conv2d_transpose_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp4assignvariableop_4_decoder_conv2d_transpose_4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp2assignvariableop_5_decoder_conv2d_transpose_4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp4assignvariableop_6_decoder_conv2d_transpose_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp2assignvariableop_7_decoder_conv2d_transpose_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8?

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?$
?
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_121340

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
C__inference_dense_6_layer_call_and_return_conditional_losses_121520

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_3_layer_call_fn_121305

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1212952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
? 
?
__inference__traced_save_121595
file_prefix5
1savev2_decoder_dense_6_kernel_read_readvariableop3
/savev2_decoder_dense_6_bias_read_readvariableop@
<savev2_decoder_conv2d_transpose_3_kernel_read_readvariableop>
:savev2_decoder_conv2d_transpose_3_bias_read_readvariableop@
<savev2_decoder_conv2d_transpose_4_kernel_read_readvariableop>
:savev2_decoder_conv2d_transpose_4_bias_read_readvariableop@
<savev2_decoder_conv2d_transpose_5_kernel_read_readvariableop>
:savev2_decoder_conv2d_transpose_5_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_decoder_dense_6_kernel_read_readvariableop/savev2_decoder_dense_6_bias_read_readvariableop<savev2_decoder_conv2d_transpose_3_kernel_read_readvariableop:savev2_decoder_conv2d_transpose_3_bias_read_readvariableop<savev2_decoder_conv2d_transpose_4_kernel_read_readvariableop:savev2_decoder_conv2d_transpose_4_bias_read_readvariableop<savev2_decoder_conv2d_transpose_5_kernel_read_readvariableop:savev2_decoder_conv2d_transpose_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*q
_input_shapes`
^: :	
?:?:@@:@: @: : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	
?:!

_output_shapes	
:?:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::	

_output_shapes
: 
?
?
3__inference_conv2d_transpose_5_layer_call_fn_121395

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_1213852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_121440

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_4_layer_call_fn_121350

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1213402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????
D
output_18
StatefulPartitionedCall:0?????????tensorflow/serving/predict:Ӏ
?
	dense
reshape
	conv1
	conv2
	conv3
regularization_losses
	variables
trainable_variables
		keras_api


signatures
E_default_save_signature
F__call__
*G&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "Decoder", "name": "decoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Decoder"}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
H__call__
*I&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 3136, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?
	variables
regularization_losses
trainable_variables
	keras_api
J__call__
*K&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 64]}}}
?


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
L__call__
*M&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 64]}}
?


kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
N__call__
*O&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 64]}}
?


!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_5", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 32]}}
 "
trackable_list_wrapper
X
0
1
2
3
4
5
!6
"7"
trackable_list_wrapper
X
0
1
2
3
4
5
!6
"7"
trackable_list_wrapper
?
'layer_metrics

(layers
)layer_regularization_losses
*metrics
regularization_losses
	variables
+non_trainable_variables
trainable_variables
F__call__
E_default_save_signature
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
,
Rserving_default"
signature_map
):'	
?2decoder/dense_6/kernel
#:!?2decoder/dense_6/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
,layer_metrics
-layer_regularization_losses
.metrics
regularization_losses

/layers
0non_trainable_variables
trainable_variables
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
1layer_metrics
2layer_regularization_losses
3metrics
regularization_losses

4layers
5non_trainable_variables
trainable_variables
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
;:9@@2!decoder/conv2d_transpose_3/kernel
-:+@2decoder/conv2d_transpose_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
6layer_metrics
7layer_regularization_losses
8metrics
regularization_losses

9layers
:non_trainable_variables
trainable_variables
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
;:9 @2!decoder/conv2d_transpose_4/kernel
-:+ 2decoder/conv2d_transpose_4/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
;layer_metrics
<layer_regularization_losses
=metrics
regularization_losses

>layers
?non_trainable_variables
trainable_variables
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
;:9 2!decoder/conv2d_transpose_5/kernel
-:+2decoder/conv2d_transpose_5/bias
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?
#	variables
@layer_metrics
Alayer_regularization_losses
Bmetrics
$regularization_losses

Clayers
Dnon_trainable_variables
%trainable_variables
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
!__inference__wrapped_model_121260?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????

?2?
(__inference_decoder_layer_call_fn_121486?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????

?2?
C__inference_decoder_layer_call_and_return_conditional_losses_121464?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????

?2?
(__inference_dense_6_layer_call_fn_121529?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_6_layer_call_and_return_conditional_losses_121520?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_reshape_1_layer_call_fn_121548?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_reshape_1_layer_call_and_return_conditional_losses_121543?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_conv2d_transpose_3_layer_call_fn_121305?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_121295?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
3__inference_conv2d_transpose_4_layer_call_fn_121350?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_121340?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
3__inference_conv2d_transpose_5_layer_call_fn_121395?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_121385?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?B?
$__inference_signature_wrapper_121509input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_121260y!"0?-
&?#
!?
input_1?????????

? ";?8
6
output_1*?'
output_1??????????
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_121295?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
3__inference_conv2d_transpose_3_layer_call_fn_121305?I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_121340?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
3__inference_conv2d_transpose_4_layer_call_fn_121350?I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_121385?!"I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
3__inference_conv2d_transpose_5_layer_call_fn_121395?!"I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
C__inference_decoder_layer_call_and_return_conditional_losses_121464}!"0?-
&?#
!?
input_1?????????

? "??<
5?2
0+???????????????????????????
? ?
(__inference_decoder_layer_call_fn_121486p!"0?-
&?#
!?
input_1?????????

? "2?/+????????????????????????????
C__inference_dense_6_layer_call_and_return_conditional_losses_121520]/?,
%?"
 ?
inputs?????????

? "&?#
?
0??????????
? |
(__inference_dense_6_layer_call_fn_121529P/?,
%?"
 ?
inputs?????????

? "????????????
E__inference_reshape_1_layer_call_and_return_conditional_losses_121543a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0?????????@
? ?
*__inference_reshape_1_layer_call_fn_121548T0?-
&?#
!?
inputs??????????
? " ??????????@?
$__inference_signature_wrapper_121509?!";?8
? 
1?.
,
input_1!?
input_1?????????
";?8
6
output_1*?'
output_1?????????
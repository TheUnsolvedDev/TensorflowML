╓З
л·
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
Ы
Conv2D

input"T
filter"T
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
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58┐Ж
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
r
dense_99/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_99/bias
k
!dense_99/bias/Read/ReadVariableOpReadVariableOpdense_99/bias*
_output_shapes
:
*
dtype0
|
dense_99/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
М╫
* 
shared_namedense_99/kernel
u
#dense_99/kernel/Read/ReadVariableOpReadVariableOpdense_99/kernel* 
_output_shapes
:
М╫
*
dtype0
t
conv2d_71/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:C*
shared_nameconv2d_71/bias
m
"conv2d_71/bias/Read/ReadVariableOpReadVariableOpconv2d_71/bias*
_output_shapes
:C*
dtype0
Д
conv2d_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:C*!
shared_nameconv2d_71/kernel
}
$conv2d_71/kernel/Read/ReadVariableOpReadVariableOpconv2d_71/kernel*&
_output_shapes
:C*
dtype0
Л
serving_default_input_54Placeholder*/
_output_shapes
:           *
dtype0*$
shape:           
В
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_54conv2d_71/kernelconv2d_71/biasdense_99/kerneldense_99/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *-
f(R&
$__inference_signature_wrapper_295281

NoOpNoOp
м
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ч
value▌B┌ B╙
┤
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
╚
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
О
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
ж
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
 
0
1
#2
$3*
 
0
1
#2
$3*
* 
░
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
*trace_0
+trace_1
,trace_2
-trace_3* 
6
.trace_0
/trace_1
0trace_2
1trace_3* 
* 
O
2
_variables
3_iterations
4_learning_rate
5_update_step_xla*

6serving_default* 

0
1*

0
1*
* 
У
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

<trace_0* 

=trace_0* 
`Z
VARIABLE_VALUEconv2d_71/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_71/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
С
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Ctrace_0* 

Dtrace_0* 

#0
$1*

#0
$1*
* 
У
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

Jtrace_0* 

Ktrace_0* 
_Y
VARIABLE_VALUEdense_99/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_99/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

L0
M1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

30*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
N	variables
O	keras_api
	Ptotal
	Qcount*
H
R	variables
S	keras_api
	Ttotal
	Ucount
V
_fn_kwargs*

P0
Q1*

N	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

T0
U1*

R	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ю
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_71/kernel/Read/ReadVariableOp"conv2d_71/bias/Read/ReadVariableOp#dense_99/kernel/Read/ReadVariableOp!dense_99/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *(
f#R!
__inference__traced_save_295451
б
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_71/kernelconv2d_71/biasdense_99/kerneldense_99/bias	iterationlearning_ratetotal_1count_1totalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *+
f&R$
"__inference__traced_restore_295491з╩
Е
Д
!__inference__wrapped_model_295093
input_54K
1model_53_conv2d_71_conv2d_readvariableop_resource:C@
2model_53_conv2d_71_biasadd_readvariableop_resource:CD
0model_53_dense_99_matmul_readvariableop_resource:
М╫
?
1model_53_dense_99_biasadd_readvariableop_resource:

identityИв)model_53/conv2d_71/BiasAdd/ReadVariableOpв(model_53/conv2d_71/Conv2D/ReadVariableOpв(model_53/dense_99/BiasAdd/ReadVariableOpв'model_53/dense_99/MatMul/ReadVariableOpв
(model_53/conv2d_71/Conv2D/ReadVariableOpReadVariableOp1model_53_conv2d_71_conv2d_readvariableop_resource*&
_output_shapes
:C*
dtype0┬
model_53/conv2d_71/Conv2DConv2Dinput_540model_53/conv2d_71/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         C*
paddingVALID*
strides
Ш
)model_53/conv2d_71/BiasAdd/ReadVariableOpReadVariableOp2model_53_conv2d_71_biasadd_readvariableop_resource*
_output_shapes
:C*
dtype0╢
model_53/conv2d_71/BiasAddBiasAdd"model_53/conv2d_71/Conv2D:output:01model_53/conv2d_71/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         C~
model_53/conv2d_71/ReluRelu#model_53/conv2d_71/BiasAdd:output:0*
T0*/
_output_shapes
:         Cj
model_53/flatten_53/ConstConst*
_output_shapes
:*
dtype0*
valueB"    Мы  е
model_53/flatten_53/ReshapeReshape%model_53/conv2d_71/Relu:activations:0"model_53/flatten_53/Const:output:0*
T0*)
_output_shapes
:         М╫Ъ
'model_53/dense_99/MatMul/ReadVariableOpReadVariableOp0model_53_dense_99_matmul_readvariableop_resource* 
_output_shapes
:
М╫
*
dtype0л
model_53/dense_99/MatMulMatMul$model_53/flatten_53/Reshape:output:0/model_53/dense_99/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
Ц
(model_53/dense_99/BiasAdd/ReadVariableOpReadVariableOp1model_53_dense_99_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0м
model_53/dense_99/BiasAddBiasAdd"model_53/dense_99/MatMul:product:00model_53/dense_99/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
z
model_53/dense_99/SoftmaxSoftmax"model_53/dense_99/BiasAdd:output:0*
T0*'
_output_shapes
:         
r
IdentityIdentity#model_53/dense_99/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
Є
NoOpNoOp*^model_53/conv2d_71/BiasAdd/ReadVariableOp)^model_53/conv2d_71/Conv2D/ReadVariableOp)^model_53/dense_99/BiasAdd/ReadVariableOp(^model_53/dense_99/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:           : : : : 2V
)model_53/conv2d_71/BiasAdd/ReadVariableOp)model_53/conv2d_71/BiasAdd/ReadVariableOp2T
(model_53/conv2d_71/Conv2D/ReadVariableOp(model_53/conv2d_71/Conv2D/ReadVariableOp2T
(model_53/dense_99/BiasAdd/ReadVariableOp(model_53/dense_99/BiasAdd/ReadVariableOp2R
'model_53/dense_99/MatMul/ReadVariableOp'model_53/dense_99/MatMul/ReadVariableOp:Y U
/
_output_shapes
:           
"
_user_specified_name
input_54
Ё-
╫
"__inference__traced_restore_295491
file_prefix;
!assignvariableop_conv2d_71_kernel:C/
!assignvariableop_1_conv2d_71_bias:C6
"assignvariableop_2_dense_99_kernel:
М╫
.
 assignvariableop_3_dense_99_bias:
&
assignvariableop_4_iteration:	 *
 assignvariableop_5_learning_rate: $
assignvariableop_6_total_1: $
assignvariableop_7_count_1: "
assignvariableop_8_total: "
assignvariableop_9_count: 
identity_11ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9и
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╬
value─B┴B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЖ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ╒
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_71_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_71_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_99_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_99_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:│
AssignVariableOp_4AssignVariableOpassignvariableop_4_iterationIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_5AssignVariableOp assignvariableop_5_learning_rateIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_6AssignVariableOpassignvariableop_6_total_1Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_7AssignVariableOpassignvariableop_7_count_1Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 л
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: Ш
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Є
▌
D__inference_model_53_layer_call_and_return_conditional_losses_295327

inputsB
(conv2d_71_conv2d_readvariableop_resource:C7
)conv2d_71_biasadd_readvariableop_resource:C;
'dense_99_matmul_readvariableop_resource:
М╫
6
(dense_99_biasadd_readvariableop_resource:

identityИв conv2d_71/BiasAdd/ReadVariableOpвconv2d_71/Conv2D/ReadVariableOpвdense_99/BiasAdd/ReadVariableOpвdense_99/MatMul/ReadVariableOpР
conv2d_71/Conv2D/ReadVariableOpReadVariableOp(conv2d_71_conv2d_readvariableop_resource*&
_output_shapes
:C*
dtype0о
conv2d_71/Conv2DConv2Dinputs'conv2d_71/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         C*
paddingVALID*
strides
Ж
 conv2d_71/BiasAdd/ReadVariableOpReadVariableOp)conv2d_71_biasadd_readvariableop_resource*
_output_shapes
:C*
dtype0Ы
conv2d_71/BiasAddBiasAddconv2d_71/Conv2D:output:0(conv2d_71/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         Cl
conv2d_71/ReluReluconv2d_71/BiasAdd:output:0*
T0*/
_output_shapes
:         Ca
flatten_53/ConstConst*
_output_shapes
:*
dtype0*
valueB"    Мы  К
flatten_53/ReshapeReshapeconv2d_71/Relu:activations:0flatten_53/Const:output:0*
T0*)
_output_shapes
:         М╫И
dense_99/MatMul/ReadVariableOpReadVariableOp'dense_99_matmul_readvariableop_resource* 
_output_shapes
:
М╫
*
dtype0Р
dense_99/MatMulMatMulflatten_53/Reshape:output:0&dense_99/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
Д
dense_99/BiasAdd/ReadVariableOpReadVariableOp(dense_99_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0С
dense_99/BiasAddBiasAdddense_99/MatMul:product:0'dense_99/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
h
dense_99/SoftmaxSoftmaxdense_99/BiasAdd:output:0*
T0*'
_output_shapes
:         
i
IdentityIdentitydense_99/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
╬
NoOpNoOp!^conv2d_71/BiasAdd/ReadVariableOp ^conv2d_71/Conv2D/ReadVariableOp ^dense_99/BiasAdd/ReadVariableOp^dense_99/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:           : : : : 2D
 conv2d_71/BiasAdd/ReadVariableOp conv2d_71/BiasAdd/ReadVariableOp2B
conv2d_71/Conv2D/ReadVariableOpconv2d_71/Conv2D/ReadVariableOp2B
dense_99/BiasAdd/ReadVariableOpdense_99/BiasAdd/ReadVariableOp2@
dense_99/MatMul/ReadVariableOpdense_99/MatMul/ReadVariableOp:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
═
Ш
)__inference_dense_99_layer_call_fn_295387

inputs
unknown:
М╫

	unknown_0:

identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dense_99_layer_call_and_return_conditional_losses_295136o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         М╫: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         М╫
 
_user_specified_nameinputs
╩
b
F__inference_flatten_53_layer_call_and_return_conditional_losses_295378

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    Мы  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         М╫Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         М╫"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         C:W S
/
_output_shapes
:         C
 
_user_specified_nameinputs
╡
╓
)__inference_model_53_layer_call_fn_295307

inputs!
unknown:C
	unknown_0:C
	unknown_1:
М╫

	unknown_2:

identityИвStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_model_53_layer_call_and_return_conditional_losses_295210o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
▄
Ю
__inference__traced_save_295451
file_prefix/
+savev2_conv2d_71_kernel_read_readvariableop-
)savev2_conv2d_71_bias_read_readvariableop.
*savev2_dense_99_kernel_read_readvariableop,
(savev2_dense_99_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: е
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╬
value─B┴B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHГ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ь
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_71_kernel_read_readvariableop)savev2_conv2d_71_bias_read_readvariableop*savev2_dense_99_kernel_read_readvariableop(savev2_dense_99_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*M
_input_shapes<
:: :C:C:
М╫
:
: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:C: 

_output_shapes
:C:&"
 
_output_shapes
:
М╫
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
▓
║
D__inference_model_53_layer_call_and_return_conditional_losses_295143

inputs*
conv2d_71_295112:C
conv2d_71_295114:C#
dense_99_295137:
М╫

dense_99_295139:

identityИв!conv2d_71/StatefulPartitionedCallв dense_99/StatefulPartitionedCallБ
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_71_295112conv2d_71_295114*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         C*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_71_layer_call_and_return_conditional_losses_295111ч
flatten_53/PartitionedCallPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         М╫* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_flatten_53_layer_call_and_return_conditional_losses_295123Т
 dense_99/StatefulPartitionedCallStatefulPartitionedCall#flatten_53/PartitionedCall:output:0dense_99_295137dense_99_295139*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dense_99_layer_call_and_return_conditional_losses_295136x
IdentityIdentity)dense_99/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
Н
NoOpNoOp"^conv2d_71/StatefulPartitionedCall!^dense_99/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:           : : : : 2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2D
 dense_99/StatefulPartitionedCall dense_99/StatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
╕
╝
D__inference_model_53_layer_call_and_return_conditional_losses_295249
input_54*
conv2d_71_295237:C
conv2d_71_295239:C#
dense_99_295243:
М╫

dense_99_295245:

identityИв!conv2d_71/StatefulPartitionedCallв dense_99/StatefulPartitionedCallГ
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCallinput_54conv2d_71_295237conv2d_71_295239*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         C*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_71_layer_call_and_return_conditional_losses_295111ч
flatten_53/PartitionedCallPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         М╫* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_flatten_53_layer_call_and_return_conditional_losses_295123Т
 dense_99/StatefulPartitionedCallStatefulPartitionedCall#flatten_53/PartitionedCall:output:0dense_99_295243dense_99_295245*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dense_99_layer_call_and_return_conditional_losses_295136x
IdentityIdentity)dense_99/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
Н
NoOpNoOp"^conv2d_71/StatefulPartitionedCall!^dense_99/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:           : : : : 2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2D
 dense_99/StatefulPartitionedCall dense_99/StatefulPartitionedCall:Y U
/
_output_shapes
:           
"
_user_specified_name
input_54
ё
Я
*__inference_conv2d_71_layer_call_fn_295356

inputs!
unknown:C
	unknown_0:C
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         C*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_71_layer_call_and_return_conditional_losses_295111w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         C`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:           : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
Є
▌
D__inference_model_53_layer_call_and_return_conditional_losses_295347

inputsB
(conv2d_71_conv2d_readvariableop_resource:C7
)conv2d_71_biasadd_readvariableop_resource:C;
'dense_99_matmul_readvariableop_resource:
М╫
6
(dense_99_biasadd_readvariableop_resource:

identityИв conv2d_71/BiasAdd/ReadVariableOpвconv2d_71/Conv2D/ReadVariableOpвdense_99/BiasAdd/ReadVariableOpвdense_99/MatMul/ReadVariableOpР
conv2d_71/Conv2D/ReadVariableOpReadVariableOp(conv2d_71_conv2d_readvariableop_resource*&
_output_shapes
:C*
dtype0о
conv2d_71/Conv2DConv2Dinputs'conv2d_71/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         C*
paddingVALID*
strides
Ж
 conv2d_71/BiasAdd/ReadVariableOpReadVariableOp)conv2d_71_biasadd_readvariableop_resource*
_output_shapes
:C*
dtype0Ы
conv2d_71/BiasAddBiasAddconv2d_71/Conv2D:output:0(conv2d_71/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         Cl
conv2d_71/ReluReluconv2d_71/BiasAdd:output:0*
T0*/
_output_shapes
:         Ca
flatten_53/ConstConst*
_output_shapes
:*
dtype0*
valueB"    Мы  К
flatten_53/ReshapeReshapeconv2d_71/Relu:activations:0flatten_53/Const:output:0*
T0*)
_output_shapes
:         М╫И
dense_99/MatMul/ReadVariableOpReadVariableOp'dense_99_matmul_readvariableop_resource* 
_output_shapes
:
М╫
*
dtype0Р
dense_99/MatMulMatMulflatten_53/Reshape:output:0&dense_99/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
Д
dense_99/BiasAdd/ReadVariableOpReadVariableOp(dense_99_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0С
dense_99/BiasAddBiasAdddense_99/MatMul:product:0'dense_99/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
h
dense_99/SoftmaxSoftmaxdense_99/BiasAdd:output:0*
T0*'
_output_shapes
:         
i
IdentityIdentitydense_99/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
╬
NoOpNoOp!^conv2d_71/BiasAdd/ReadVariableOp ^conv2d_71/Conv2D/ReadVariableOp ^dense_99/BiasAdd/ReadVariableOp^dense_99/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:           : : : : 2D
 conv2d_71/BiasAdd/ReadVariableOp conv2d_71/BiasAdd/ReadVariableOp2B
conv2d_71/Conv2D/ReadVariableOpconv2d_71/Conv2D/ReadVariableOp2B
dense_99/BiasAdd/ReadVariableOpdense_99/BiasAdd/ReadVariableOp2@
dense_99/MatMul/ReadVariableOpdense_99/MatMul/ReadVariableOp:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
╕
╝
D__inference_model_53_layer_call_and_return_conditional_losses_295264
input_54*
conv2d_71_295252:C
conv2d_71_295254:C#
dense_99_295258:
М╫

dense_99_295260:

identityИв!conv2d_71/StatefulPartitionedCallв dense_99/StatefulPartitionedCallГ
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCallinput_54conv2d_71_295252conv2d_71_295254*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         C*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_71_layer_call_and_return_conditional_losses_295111ч
flatten_53/PartitionedCallPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         М╫* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_flatten_53_layer_call_and_return_conditional_losses_295123Т
 dense_99/StatefulPartitionedCallStatefulPartitionedCall#flatten_53/PartitionedCall:output:0dense_99_295258dense_99_295260*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dense_99_layer_call_and_return_conditional_losses_295136x
IdentityIdentity)dense_99/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
Н
NoOpNoOp"^conv2d_71/StatefulPartitionedCall!^dense_99/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:           : : : : 2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2D
 dense_99/StatefulPartitionedCall dense_99/StatefulPartitionedCall:Y U
/
_output_shapes
:           
"
_user_specified_name
input_54
║
G
+__inference_flatten_53_layer_call_fn_295372

inputs
identity╕
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         М╫* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_flatten_53_layer_call_and_return_conditional_losses_295123b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         М╫"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         C:W S
/
_output_shapes
:         C
 
_user_specified_nameinputs
╗
╪
)__inference_model_53_layer_call_fn_295234
input_54!
unknown:C
	unknown_0:C
	unknown_1:
М╫

	unknown_2:

identityИвStatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinput_54unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_model_53_layer_call_and_return_conditional_losses_295210o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:           
"
_user_specified_name
input_54
▓
║
D__inference_model_53_layer_call_and_return_conditional_losses_295210

inputs*
conv2d_71_295198:C
conv2d_71_295200:C#
dense_99_295204:
М╫

dense_99_295206:

identityИв!conv2d_71/StatefulPartitionedCallв dense_99/StatefulPartitionedCallБ
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_71_295198conv2d_71_295200*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         C*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_71_layer_call_and_return_conditional_losses_295111ч
flatten_53/PartitionedCallPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         М╫* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_flatten_53_layer_call_and_return_conditional_losses_295123Т
 dense_99/StatefulPartitionedCallStatefulPartitionedCall#flatten_53/PartitionedCall:output:0dense_99_295204dense_99_295206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dense_99_layer_call_and_return_conditional_losses_295136x
IdentityIdentity)dense_99/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
Н
NoOpNoOp"^conv2d_71/StatefulPartitionedCall!^dense_99/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:           : : : : 2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2D
 dense_99/StatefulPartitionedCall dense_99/StatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
и

ў
D__inference_dense_99_layer_call_and_return_conditional_losses_295136

inputs2
matmul_readvariableop_resource:
М╫
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
М╫
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         М╫: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:         М╫
 
_user_specified_nameinputs
╡
╓
)__inference_model_53_layer_call_fn_295294

inputs!
unknown:C
	unknown_0:C
	unknown_1:
М╫

	unknown_2:

identityИвStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_model_53_layer_call_and_return_conditional_losses_295143o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
╗
╪
)__inference_model_53_layer_call_fn_295154
input_54!
unknown:C
	unknown_0:C
	unknown_1:
М╫

	unknown_2:

identityИвStatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinput_54unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_model_53_layer_call_and_return_conditional_losses_295143o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:           
"
_user_specified_name
input_54
У
╙
$__inference_signature_wrapper_295281
input_54!
unknown:C
	unknown_0:C
	unknown_1:
М╫

	unknown_2:

identityИвStatefulPartitionedCall╫
StatefulPartitionedCallStatefulPartitionedCallinput_54unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В **
f%R#
!__inference__wrapped_model_295093o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:           
"
_user_specified_name
input_54
Е
■
E__inference_conv2d_71_layer_call_and_return_conditional_losses_295367

inputs8
conv2d_readvariableop_resource:C-
biasadd_readvariableop_resource:C
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:C*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         C*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:C*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         CX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         Ci
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         Cw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
╩
b
F__inference_flatten_53_layer_call_and_return_conditional_losses_295123

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    Мы  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         М╫Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         М╫"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         C:W S
/
_output_shapes
:         C
 
_user_specified_nameinputs
и

ў
D__inference_dense_99_layer_call_and_return_conditional_losses_295398

inputs2
matmul_readvariableop_resource:
М╫
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
М╫
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         М╫: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:         М╫
 
_user_specified_nameinputs
Е
■
E__inference_conv2d_71_layer_call_and_return_conditional_losses_295111

inputs8
conv2d_readvariableop_resource:C-
biasadd_readvariableop_resource:C
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:C*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         C*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:C*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         CX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         Ci
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         Cw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:           
 
_user_specified_nameinputs"Ж
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╡
serving_defaultб
E
input_549
serving_default_input_54:0           <
dense_990
StatefulPartitionedCall:0         
tensorflow/serving/predict:√l
╦
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
▌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
е
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
<
0
1
#2
$3"
trackable_list_wrapper
<
0
1
#2
$3"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
┘
*trace_0
+trace_1
,trace_2
-trace_32ю
)__inference_model_53_layer_call_fn_295154
)__inference_model_53_layer_call_fn_295294
)__inference_model_53_layer_call_fn_295307
)__inference_model_53_layer_call_fn_295234┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z*trace_0z+trace_1z,trace_2z-trace_3
┼
.trace_0
/trace_1
0trace_2
1trace_32┌
D__inference_model_53_layer_call_and_return_conditional_losses_295327
D__inference_model_53_layer_call_and_return_conditional_losses_295347
D__inference_model_53_layer_call_and_return_conditional_losses_295249
D__inference_model_53_layer_call_and_return_conditional_losses_295264┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z.trace_0z/trace_1z0trace_2z1trace_3
═B╩
!__inference__wrapped_model_295093input_54"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
j
2
_variables
3_iterations
4_learning_rate
5_update_step_xla"
experimentalOptimizer
,
6serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ю
<trace_02╤
*__inference_conv2d_71_layer_call_fn_295356в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z<trace_0
Й
=trace_02ь
E__inference_conv2d_71_layer_call_and_return_conditional_losses_295367в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z=trace_0
*:(C2conv2d_71/kernel
:C2conv2d_71/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
я
Ctrace_02╥
+__inference_flatten_53_layer_call_fn_295372в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zCtrace_0
К
Dtrace_02э
F__inference_flatten_53_layer_call_and_return_conditional_losses_295378в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zDtrace_0
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
э
Jtrace_02╨
)__inference_dense_99_layer_call_fn_295387в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zJtrace_0
И
Ktrace_02ы
D__inference_dense_99_layer_call_and_return_conditional_losses_295398в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zKtrace_0
#:!
М╫
2dense_99/kernel
:
2dense_99/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№B∙
)__inference_model_53_layer_call_fn_295154input_54"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
)__inference_model_53_layer_call_fn_295294inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
)__inference_model_53_layer_call_fn_295307inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
)__inference_model_53_layer_call_fn_295234input_54"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
D__inference_model_53_layer_call_and_return_conditional_losses_295327inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
D__inference_model_53_layer_call_and_return_conditional_losses_295347inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
D__inference_model_53_layer_call_and_return_conditional_losses_295249input_54"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
D__inference_model_53_layer_call_and_return_conditional_losses_295264input_54"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
'
30"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
┐2╝╣
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
╠B╔
$__inference_signature_wrapper_295281input_54"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
▐B█
*__inference_conv2d_71_layer_call_fn_295356inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_conv2d_71_layer_call_and_return_conditional_losses_295367inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
▀B▄
+__inference_flatten_53_layer_call_fn_295372inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_flatten_53_layer_call_and_return_conditional_losses_295378inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
▌B┌
)__inference_dense_99_layer_call_fn_295387inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_dense_99_layer_call_and_return_conditional_losses_295398inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
N
N	variables
O	keras_api
	Ptotal
	Qcount"
_tf_keras_metric
^
R	variables
S	keras_api
	Ttotal
	Ucount
V
_fn_kwargs"
_tf_keras_metric
.
P0
Q1"
trackable_list_wrapper
-
N	variables"
_generic_user_object
:  (2total
:  (2count
.
T0
U1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperЫ
!__inference__wrapped_model_295093v#$9в6
/в,
*К'
input_54           
к "3к0
.
dense_99"К
dense_99         
╝
E__inference_conv2d_71_layer_call_and_return_conditional_losses_295367s7в4
-в*
(К%
inputs           
к "4в1
*К'
tensor_0         C
Ъ Ц
*__inference_conv2d_71_layer_call_fn_295356h7в4
-в*
(К%
inputs           
к ")К&
unknown         Cн
D__inference_dense_99_layer_call_and_return_conditional_losses_295398e#$1в.
'в$
"К
inputs         М╫
к ",в)
"К
tensor_0         

Ъ З
)__inference_dense_99_layer_call_fn_295387Z#$1в.
'в$
"К
inputs         М╫
к "!К
unknown         
│
F__inference_flatten_53_layer_call_and_return_conditional_losses_295378i7в4
-в*
(К%
inputs         C
к ".в+
$К!
tensor_0         М╫
Ъ Н
+__inference_flatten_53_layer_call_fn_295372^7в4
-в*
(К%
inputs         C
к "#К 
unknown         М╫┐
D__inference_model_53_layer_call_and_return_conditional_losses_295249w#$Aв>
7в4
*К'
input_54           
p 

 
к ",в)
"К
tensor_0         

Ъ ┐
D__inference_model_53_layer_call_and_return_conditional_losses_295264w#$Aв>
7в4
*К'
input_54           
p

 
к ",в)
"К
tensor_0         

Ъ ╜
D__inference_model_53_layer_call_and_return_conditional_losses_295327u#$?в<
5в2
(К%
inputs           
p 

 
к ",в)
"К
tensor_0         

Ъ ╜
D__inference_model_53_layer_call_and_return_conditional_losses_295347u#$?в<
5в2
(К%
inputs           
p

 
к ",в)
"К
tensor_0         

Ъ Щ
)__inference_model_53_layer_call_fn_295154l#$Aв>
7в4
*К'
input_54           
p 

 
к "!К
unknown         
Щ
)__inference_model_53_layer_call_fn_295234l#$Aв>
7в4
*К'
input_54           
p

 
к "!К
unknown         
Ч
)__inference_model_53_layer_call_fn_295294j#$?в<
5в2
(К%
inputs           
p 

 
к "!К
unknown         
Ч
)__inference_model_53_layer_call_fn_295307j#$?в<
5в2
(К%
inputs           
p

 
к "!К
unknown         
л
$__inference_signature_wrapper_295281В#$EвB
в 
;к8
6
input_54*К'
input_54           "3к0
.
dense_99"К
dense_99         

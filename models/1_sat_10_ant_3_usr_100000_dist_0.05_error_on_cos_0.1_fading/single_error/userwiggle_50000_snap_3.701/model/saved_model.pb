��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	
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
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
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
�
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
executor_typestring ��
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
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.12v2.10.0-76-gfdfc646704c8��
�
Adam_2/dense_25/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:<**
shared_nameAdam_2/dense_25/bias/vhat
�
-Adam_2/dense_25/bias/vhat/Read/ReadVariableOpReadVariableOpAdam_2/dense_25/bias/vhat*
_output_shapes
:<*
dtype0
�
Adam_2/dense_25/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�<*,
shared_nameAdam_2/dense_25/kernel/vhat
�
/Adam_2/dense_25/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam_2/dense_25/kernel/vhat*
_output_shapes
:	�<*
dtype0
�
Adam_2/dense_24/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:<**
shared_nameAdam_2/dense_24/bias/vhat
�
-Adam_2/dense_24/bias/vhat/Read/ReadVariableOpReadVariableOpAdam_2/dense_24/bias/vhat*
_output_shapes
:<*
dtype0
�
Adam_2/dense_24/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�<*,
shared_nameAdam_2/dense_24/kernel/vhat
�
/Adam_2/dense_24/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam_2/dense_24/kernel/vhat*
_output_shapes
:	�<*
dtype0
�
Adam_2/dense_23/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameAdam_2/dense_23/bias/vhat
�
-Adam_2/dense_23/bias/vhat/Read/ReadVariableOpReadVariableOpAdam_2/dense_23/bias/vhat*
_output_shapes	
:�*
dtype0
�
Adam_2/dense_23/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*,
shared_nameAdam_2/dense_23/kernel/vhat
�
/Adam_2/dense_23/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam_2/dense_23/kernel/vhat* 
_output_shapes
:
��*
dtype0
�
Adam_2/dense_22/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameAdam_2/dense_22/bias/vhat
�
-Adam_2/dense_22/bias/vhat/Read/ReadVariableOpReadVariableOpAdam_2/dense_22/bias/vhat*
_output_shapes	
:�*
dtype0
�
Adam_2/dense_22/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*,
shared_nameAdam_2/dense_22/kernel/vhat
�
/Adam_2/dense_22/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam_2/dense_22/kernel/vhat* 
_output_shapes
:
��*
dtype0
�
Adam_2/dense_21/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameAdam_2/dense_21/bias/vhat
�
-Adam_2/dense_21/bias/vhat/Read/ReadVariableOpReadVariableOpAdam_2/dense_21/bias/vhat*
_output_shapes	
:�*
dtype0
�
Adam_2/dense_21/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*,
shared_nameAdam_2/dense_21/kernel/vhat
�
/Adam_2/dense_21/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam_2/dense_21/kernel/vhat* 
_output_shapes
:
��*
dtype0
�
Adam_2/dense_20/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameAdam_2/dense_20/bias/vhat
�
-Adam_2/dense_20/bias/vhat/Read/ReadVariableOpReadVariableOpAdam_2/dense_20/bias/vhat*
_output_shapes	
:�*
dtype0
�
Adam_2/dense_20/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<�*,
shared_nameAdam_2/dense_20/kernel/vhat
�
/Adam_2/dense_20/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam_2/dense_20/kernel/vhat*
_output_shapes
:	<�*
dtype0
�
Adam_2/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*'
shared_nameAdam_2/dense_25/bias/v
}
*Adam_2/dense_25/bias/v/Read/ReadVariableOpReadVariableOpAdam_2/dense_25/bias/v*
_output_shapes
:<*
dtype0
�
Adam_2/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�<*)
shared_nameAdam_2/dense_25/kernel/v
�
,Adam_2/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam_2/dense_25/kernel/v*
_output_shapes
:	�<*
dtype0
�
Adam_2/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*'
shared_nameAdam_2/dense_24/bias/v
}
*Adam_2/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam_2/dense_24/bias/v*
_output_shapes
:<*
dtype0
�
Adam_2/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�<*)
shared_nameAdam_2/dense_24/kernel/v
�
,Adam_2/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam_2/dense_24/kernel/v*
_output_shapes
:	�<*
dtype0
�
Adam_2/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam_2/dense_23/bias/v
~
*Adam_2/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam_2/dense_23/bias/v*
_output_shapes	
:�*
dtype0
�
Adam_2/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameAdam_2/dense_23/kernel/v
�
,Adam_2/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam_2/dense_23/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam_2/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam_2/dense_22/bias/v
~
*Adam_2/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam_2/dense_22/bias/v*
_output_shapes	
:�*
dtype0
�
Adam_2/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameAdam_2/dense_22/kernel/v
�
,Adam_2/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam_2/dense_22/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam_2/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam_2/dense_21/bias/v
~
*Adam_2/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam_2/dense_21/bias/v*
_output_shapes	
:�*
dtype0
�
Adam_2/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameAdam_2/dense_21/kernel/v
�
,Adam_2/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam_2/dense_21/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam_2/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam_2/dense_20/bias/v
~
*Adam_2/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdam_2/dense_20/bias/v*
_output_shapes	
:�*
dtype0
�
Adam_2/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<�*)
shared_nameAdam_2/dense_20/kernel/v
�
,Adam_2/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam_2/dense_20/kernel/v*
_output_shapes
:	<�*
dtype0
�
Adam_2/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*'
shared_nameAdam_2/dense_25/bias/m
}
*Adam_2/dense_25/bias/m/Read/ReadVariableOpReadVariableOpAdam_2/dense_25/bias/m*
_output_shapes
:<*
dtype0
�
Adam_2/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�<*)
shared_nameAdam_2/dense_25/kernel/m
�
,Adam_2/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam_2/dense_25/kernel/m*
_output_shapes
:	�<*
dtype0
�
Adam_2/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*'
shared_nameAdam_2/dense_24/bias/m
}
*Adam_2/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam_2/dense_24/bias/m*
_output_shapes
:<*
dtype0
�
Adam_2/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�<*)
shared_nameAdam_2/dense_24/kernel/m
�
,Adam_2/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam_2/dense_24/kernel/m*
_output_shapes
:	�<*
dtype0
�
Adam_2/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam_2/dense_23/bias/m
~
*Adam_2/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam_2/dense_23/bias/m*
_output_shapes	
:�*
dtype0
�
Adam_2/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameAdam_2/dense_23/kernel/m
�
,Adam_2/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam_2/dense_23/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam_2/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam_2/dense_22/bias/m
~
*Adam_2/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam_2/dense_22/bias/m*
_output_shapes	
:�*
dtype0
�
Adam_2/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameAdam_2/dense_22/kernel/m
�
,Adam_2/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam_2/dense_22/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam_2/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam_2/dense_21/bias/m
~
*Adam_2/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam_2/dense_21/bias/m*
_output_shapes	
:�*
dtype0
�
Adam_2/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameAdam_2/dense_21/kernel/m
�
,Adam_2/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam_2/dense_21/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam_2/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam_2/dense_20/bias/m
~
*Adam_2/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdam_2/dense_20/bias/m*
_output_shapes	
:�*
dtype0
�
Adam_2/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<�*)
shared_nameAdam_2/dense_20/kernel/m
�
,Adam_2/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam_2/dense_20/kernel/m*
_output_shapes
:	<�*
dtype0
\
iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameiter
U
iter/Read/ReadVariableOpReadVariableOpiter*
_output_shapes
: *
dtype0	
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
r
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*
shared_namedense_25/bias
k
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes
:<*
dtype0
{
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�<* 
shared_namedense_25/kernel
t
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
_output_shapes
:	�<*
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:<*
dtype0
{
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�<* 
shared_namedense_24/kernel
t
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes
:	�<*
dtype0
�
&batch_normalization_19/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_19/moving_variance
�
:batch_normalization_19/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_19/moving_variance*
_output_shapes	
:�*
dtype0
�
"batch_normalization_19/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_19/moving_mean
�
6batch_normalization_19/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_19/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_18/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_18/moving_variance
�
:batch_normalization_18/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_18/moving_variance*
_output_shapes	
:�*
dtype0
�
"batch_normalization_18/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_18/moving_mean
�
6batch_normalization_18/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_18/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_17/moving_variance
�
:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes	
:�*
dtype0
�
"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_17/moving_mean
�
6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_16/moving_variance
�
:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes	
:�*
dtype0
�
"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_16/moving_mean
�
6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes	
:�*
dtype0
s
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_23/bias
l
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes	
:�*
dtype0
|
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_23/kernel
u
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel* 
_output_shapes
:
��*
dtype0
s
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_22/bias
l
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes	
:�*
dtype0
|
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_22/kernel
u
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel* 
_output_shapes
:
��*
dtype0
s
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_21/bias
l
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes	
:�*
dtype0
|
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_21/kernel
u
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel* 
_output_shapes
:
��*
dtype0
s
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_20/bias
l
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes	
:�*
dtype0
{
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<�* 
shared_namedense_20/kernel
t
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes
:	<�*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������<*
dtype0*
shape:���������<
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_20/kerneldense_20/bias"batch_normalization_16/moving_mean&batch_normalization_16/moving_variancedense_21/kerneldense_21/bias"batch_normalization_17/moving_mean&batch_normalization_17/moving_variancedense_22/kerneldense_22/bias"batch_normalization_18/moving_mean&batch_normalization_18/moving_variancedense_23/kerneldense_23/bias"batch_normalization_19/moving_mean&batch_normalization_19/moving_variancedense_24/kerneldense_24/biasdense_25/kerneldense_25/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������<:���������<*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_15807906

NoOpNoOp
�o
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�n
value�nB�n B�n
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
hidden_layers
	batch_norm_layers

output_layer_means
output_layer_log_stds
	optimizer
loss
call
#get_action_and_log_prob_density

signatures*
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15
!16
"17
#18
$19*
Z
0
1
2
3
4
5
6
7
!8
"9
#10
$11*
* 
�
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
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
 
20
31
42
53*
 
60
71
82
93*
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

!kernel
"bias*
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

#kernel
$bias*
�

Fbeta_1

Gbeta_2
	Hdecay
Iiterm�m�m�m�m�m�m�m�!m�"m�#m�$m�v�v�v�v�v�v�v�v�!v�"v�#v�$v�vhat�vhat�vhat�vhat�vhat�vhat�vhat�vhat�!vhat�"vhat�#vhat�$vhat�*
* 
C
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_3
Ntrace_4* 

Otrace_0
Ptrace_1* 

Qserving_default* 
OI
VARIABLE_VALUEdense_20/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_20/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_21/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_21/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_22/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_22/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_23/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_23/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"batch_normalization_16/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_normalization_16/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"batch_normalization_17/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&batch_normalization_17/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"batch_normalization_18/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&batch_normalization_18/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"batch_normalization_19/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&batch_normalization_19/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_24/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_24/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_25/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_25/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
<
0
1
2
3
4
5
6
 7*
J
20
31
42
53
64
75
86
97

8
9*
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
�
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses

kernel
bias*
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

kernel
bias*
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

kernel
bias*
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

kernel
bias*
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
paxis
moving_mean
moving_variance*
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses
waxis
moving_mean
moving_variance*
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses
~axis
moving_mean
moving_variance*
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
moving_mean
 moving_variance*

!0
"1*

!0
"1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

#0
$1*

#0
$1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
KE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
GA
VARIABLE_VALUEiter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

0
1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

0
1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

0
 1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
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

0
1*
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 

0
 1*
* 
* 
* 
* 
* 
* 
* 
* 
tn
VARIABLE_VALUEAdam_2/dense_20/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam_2/dense_20/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam_2/dense_21/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam_2/dense_21/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam_2/dense_22/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam_2/dense_22/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam_2/dense_23/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam_2/dense_23/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam_2/dense_24/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam_2/dense_24/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam_2/dense_25/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam_2/dense_25/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam_2/dense_20/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam_2/dense_20/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam_2/dense_21/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam_2/dense_21/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam_2/dense_22/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam_2/dense_22/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam_2/dense_23/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam_2/dense_23/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam_2/dense_24/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam_2/dense_24/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam_2/dense_25/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam_2/dense_25/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam_2/dense_20/kernel/vhatEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam_2/dense_20/bias/vhatEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam_2/dense_21/kernel/vhatEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam_2/dense_21/bias/vhatEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam_2/dense_22/kernel/vhatEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam_2/dense_22/bias/vhatEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam_2/dense_23/kernel/vhatEvariables/6/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam_2/dense_23/bias/vhatEvariables/7/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam_2/dense_24/kernel/vhatFvariables/16/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam_2/dense_24/bias/vhatFvariables/17/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam_2/dense_25/kernel/vhatFvariables/18/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam_2/dense_25/bias/vhatFvariables/19/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp6batch_normalization_18/moving_mean/Read/ReadVariableOp:batch_normalization_18/moving_variance/Read/ReadVariableOp6batch_normalization_19/moving_mean/Read/ReadVariableOp:batch_normalization_19/moving_variance/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOpiter/Read/ReadVariableOp,Adam_2/dense_20/kernel/m/Read/ReadVariableOp*Adam_2/dense_20/bias/m/Read/ReadVariableOp,Adam_2/dense_21/kernel/m/Read/ReadVariableOp*Adam_2/dense_21/bias/m/Read/ReadVariableOp,Adam_2/dense_22/kernel/m/Read/ReadVariableOp*Adam_2/dense_22/bias/m/Read/ReadVariableOp,Adam_2/dense_23/kernel/m/Read/ReadVariableOp*Adam_2/dense_23/bias/m/Read/ReadVariableOp,Adam_2/dense_24/kernel/m/Read/ReadVariableOp*Adam_2/dense_24/bias/m/Read/ReadVariableOp,Adam_2/dense_25/kernel/m/Read/ReadVariableOp*Adam_2/dense_25/bias/m/Read/ReadVariableOp,Adam_2/dense_20/kernel/v/Read/ReadVariableOp*Adam_2/dense_20/bias/v/Read/ReadVariableOp,Adam_2/dense_21/kernel/v/Read/ReadVariableOp*Adam_2/dense_21/bias/v/Read/ReadVariableOp,Adam_2/dense_22/kernel/v/Read/ReadVariableOp*Adam_2/dense_22/bias/v/Read/ReadVariableOp,Adam_2/dense_23/kernel/v/Read/ReadVariableOp*Adam_2/dense_23/bias/v/Read/ReadVariableOp,Adam_2/dense_24/kernel/v/Read/ReadVariableOp*Adam_2/dense_24/bias/v/Read/ReadVariableOp,Adam_2/dense_25/kernel/v/Read/ReadVariableOp*Adam_2/dense_25/bias/v/Read/ReadVariableOp/Adam_2/dense_20/kernel/vhat/Read/ReadVariableOp-Adam_2/dense_20/bias/vhat/Read/ReadVariableOp/Adam_2/dense_21/kernel/vhat/Read/ReadVariableOp-Adam_2/dense_21/bias/vhat/Read/ReadVariableOp/Adam_2/dense_22/kernel/vhat/Read/ReadVariableOp-Adam_2/dense_22/bias/vhat/Read/ReadVariableOp/Adam_2/dense_23/kernel/vhat/Read/ReadVariableOp-Adam_2/dense_23/bias/vhat/Read/ReadVariableOp/Adam_2/dense_24/kernel/vhat/Read/ReadVariableOp-Adam_2/dense_24/bias/vhat/Read/ReadVariableOp/Adam_2/dense_25/kernel/vhat/Read/ReadVariableOp-Adam_2/dense_25/bias/vhat/Read/ReadVariableOpConst*I
TinB
@2>	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_save_15808872
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/bias"batch_normalization_16/moving_mean&batch_normalization_16/moving_variance"batch_normalization_17/moving_mean&batch_normalization_17/moving_variance"batch_normalization_18/moving_mean&batch_normalization_18/moving_variance"batch_normalization_19/moving_mean&batch_normalization_19/moving_variancedense_24/kerneldense_24/biasdense_25/kerneldense_25/biasbeta_1beta_2decayiterAdam_2/dense_20/kernel/mAdam_2/dense_20/bias/mAdam_2/dense_21/kernel/mAdam_2/dense_21/bias/mAdam_2/dense_22/kernel/mAdam_2/dense_22/bias/mAdam_2/dense_23/kernel/mAdam_2/dense_23/bias/mAdam_2/dense_24/kernel/mAdam_2/dense_24/bias/mAdam_2/dense_25/kernel/mAdam_2/dense_25/bias/mAdam_2/dense_20/kernel/vAdam_2/dense_20/bias/vAdam_2/dense_21/kernel/vAdam_2/dense_21/bias/vAdam_2/dense_22/kernel/vAdam_2/dense_22/bias/vAdam_2/dense_23/kernel/vAdam_2/dense_23/bias/vAdam_2/dense_24/kernel/vAdam_2/dense_24/bias/vAdam_2/dense_25/kernel/vAdam_2/dense_25/bias/vAdam_2/dense_20/kernel/vhatAdam_2/dense_20/bias/vhatAdam_2/dense_21/kernel/vhatAdam_2/dense_21/bias/vhatAdam_2/dense_22/kernel/vhatAdam_2/dense_22/bias/vhatAdam_2/dense_23/kernel/vhatAdam_2/dense_23/bias/vhatAdam_2/dense_24/kernel/vhatAdam_2/dense_24/bias/vhatAdam_2/dense_25/kernel/vhatAdam_2/dense_25/bias/vhat*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference__traced_restore_15809062�
��
�
__inference_call_6289

inputs:
'dense_20_matmul_readvariableop_resource:	<�7
(dense_20_biasadd_readvariableop_resource:	�B
3batch_normalization_16_cast_readvariableop_resource:	�D
5batch_normalization_16_cast_1_readvariableop_resource:	�;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�B
3batch_normalization_17_cast_readvariableop_resource:	�D
5batch_normalization_17_cast_1_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�B
3batch_normalization_18_cast_readvariableop_resource:	�D
5batch_normalization_18_cast_1_readvariableop_resource:	�;
'dense_23_matmul_readvariableop_resource:
��7
(dense_23_biasadd_readvariableop_resource:	�B
3batch_normalization_19_cast_readvariableop_resource:	�D
5batch_normalization_19_cast_1_readvariableop_resource:	�:
'dense_24_matmul_readvariableop_resource:	�<6
(dense_24_biasadd_readvariableop_resource:<:
'dense_25_matmul_readvariableop_resource:	�<6
(dense_25_biasadd_readvariableop_resource:<
identity

identity_1��*batch_normalization_16/Cast/ReadVariableOp�,batch_normalization_16/Cast_1/ReadVariableOp�*batch_normalization_17/Cast/ReadVariableOp�,batch_normalization_17/Cast_1/ReadVariableOp�*batch_normalization_18/Cast/ReadVariableOp�,batch_normalization_18/Cast_1/ReadVariableOp�*batch_normalization_19/Cast/ReadVariableOp�,batch_normalization_19/Cast_1/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	<�*
dtype0|
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0*(
_output_shapes
:����������T
dense_20/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
dense_20/LessLessdense_20/BiasAdd:output:0dense_20/Less/y:output:0*
T0*(
_output_shapes
:����������S
dense_20/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>r
dense_20/MulMuldense_20/Tanh:y:0dense_20/Mul/y:output:0*
T0*(
_output_shapes
:�����������
dense_20/SelectV2SelectV2dense_20/Less:z:0dense_20/Mul:z:0dense_20/Tanh:y:0*
T0*(
_output_shapes
:�����������
*batch_normalization_16/Cast/ReadVariableOpReadVariableOp3batch_normalization_16_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_16/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_16/batchnorm/addAddV24batch_normalization_16/Cast_1/ReadVariableOp:value:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_16/batchnorm/mulMuldense_20/SelectV2:output:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
$batch_normalization_16/batchnorm/NegNeg2batch_normalization_16/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/mul_1Mul(batch_normalization_16/batchnorm/Neg:y:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/add_1AddV2(batch_normalization_16/batchnorm/mul:z:0*batch_normalization_16/batchnorm/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMul*batch_normalization_16/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_21/TanhTanhdense_21/BiasAdd:output:0*
T0*(
_output_shapes
:����������T
dense_21/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
dense_21/LessLessdense_21/BiasAdd:output:0dense_21/Less/y:output:0*
T0*(
_output_shapes
:����������S
dense_21/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>r
dense_21/MulMuldense_21/Tanh:y:0dense_21/Mul/y:output:0*
T0*(
_output_shapes
:�����������
dense_21/SelectV2SelectV2dense_21/Less:z:0dense_21/Mul:z:0dense_21/Tanh:y:0*
T0*(
_output_shapes
:�����������
*batch_normalization_17/Cast/ReadVariableOpReadVariableOp3batch_normalization_17_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_17/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_17_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_17/batchnorm/addAddV24batch_normalization_17/Cast_1/ReadVariableOp:value:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_17/batchnorm/mulMuldense_21/SelectV2:output:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
$batch_normalization_17/batchnorm/NegNeg2batch_normalization_17/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/mul_1Mul(batch_normalization_17/batchnorm/Neg:y:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/add_1AddV2(batch_normalization_17/batchnorm/mul:z:0*batch_normalization_17/batchnorm/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_22/TanhTanhdense_22/BiasAdd:output:0*
T0*(
_output_shapes
:����������T
dense_22/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
dense_22/LessLessdense_22/BiasAdd:output:0dense_22/Less/y:output:0*
T0*(
_output_shapes
:����������S
dense_22/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>r
dense_22/MulMuldense_22/Tanh:y:0dense_22/Mul/y:output:0*
T0*(
_output_shapes
:�����������
dense_22/SelectV2SelectV2dense_22/Less:z:0dense_22/Mul:z:0dense_22/Tanh:y:0*
T0*(
_output_shapes
:�����������
*batch_normalization_18/Cast/ReadVariableOpReadVariableOp3batch_normalization_18_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_18/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_18_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_18/batchnorm/addAddV24batch_normalization_18/Cast_1/ReadVariableOp:value:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_18/batchnorm/mulMuldense_22/SelectV2:output:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
$batch_normalization_18/batchnorm/NegNeg2batch_normalization_18/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/mul_1Mul(batch_normalization_18/batchnorm/Neg:y:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/add_1AddV2(batch_normalization_18/batchnorm/mul:z:0*batch_normalization_18/batchnorm/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_23/MatMulMatMul*batch_normalization_18/batchnorm/add_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_23/TanhTanhdense_23/BiasAdd:output:0*
T0*(
_output_shapes
:����������T
dense_23/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
dense_23/LessLessdense_23/BiasAdd:output:0dense_23/Less/y:output:0*
T0*(
_output_shapes
:����������S
dense_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>r
dense_23/MulMuldense_23/Tanh:y:0dense_23/Mul/y:output:0*
T0*(
_output_shapes
:�����������
dense_23/SelectV2SelectV2dense_23/Less:z:0dense_23/Mul:z:0dense_23/Tanh:y:0*
T0*(
_output_shapes
:�����������
*batch_normalization_19/Cast/ReadVariableOpReadVariableOp3batch_normalization_19_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_19/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_19_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_19/batchnorm/addAddV24batch_normalization_19/Cast_1/ReadVariableOp:value:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_19/batchnorm/mulMuldense_23/SelectV2:output:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
$batch_normalization_19/batchnorm/NegNeg2batch_normalization_19/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/mul_1Mul(batch_normalization_19/batchnorm/Neg:y:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/add_1AddV2(batch_normalization_19/batchnorm/mul:z:0*batch_normalization_19/batchnorm/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_24/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_25/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
clip_by_value/MinimumMinimumdense_25/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������<T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������<h
IdentityIdentitydense_24/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������<b

Identity_1Identityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:���������<�
NoOpNoOp+^batch_normalization_16/Cast/ReadVariableOp-^batch_normalization_16/Cast_1/ReadVariableOp+^batch_normalization_17/Cast/ReadVariableOp-^batch_normalization_17/Cast_1/ReadVariableOp+^batch_normalization_18/Cast/ReadVariableOp-^batch_normalization_18/Cast_1/ReadVariableOp+^batch_normalization_19/Cast/ReadVariableOp-^batch_normalization_19/Cast_1/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������<: : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_16/Cast/ReadVariableOp*batch_normalization_16/Cast/ReadVariableOp2\
,batch_normalization_16/Cast_1/ReadVariableOp,batch_normalization_16/Cast_1/ReadVariableOp2X
*batch_normalization_17/Cast/ReadVariableOp*batch_normalization_17/Cast/ReadVariableOp2\
,batch_normalization_17/Cast_1/ReadVariableOp,batch_normalization_17/Cast_1/ReadVariableOp2X
*batch_normalization_18/Cast/ReadVariableOp*batch_normalization_18/Cast/ReadVariableOp2\
,batch_normalization_18/Cast_1/ReadVariableOp,batch_normalization_18/Cast_1/ReadVariableOp2X
*batch_normalization_19/Cast/ReadVariableOp*batch_normalization_19/Cast/ReadVariableOp2\
,batch_normalization_19/Cast_1/ReadVariableOp,batch_normalization_19/Cast_1/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
F__inference_dense_23_layer_call_and_return_conditional_losses_15807381

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
LessLessBiasAdd:output:0Less/y:output:0*
T0*(
_output_shapes
:����������J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>W
MulMulTanh:y:0Mul/y:output:0*
T0*(
_output_shapes
:����������d
SelectV2SelectV2Less:z:0Mul:z:0Tanh:y:0*
T0*(
_output_shapes
:����������a
IdentityIdentitySelectV2:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_19_layer_call_and_return_conditional_losses_15808668

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�d
batchnorm/mulMulinputsbatchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:����������T
batchnorm/NegNegmoments/Squeeze:output:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulbatchnorm/Neg:y:0batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul:z:0batchnorm/mul_1:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_policy_network_soft_layer_call_fn_15807475
input_1
unknown:	<�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�<

unknown_16:<

unknown_17:	�<

unknown_18:<
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������<:���������<*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15807430o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������<: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������<
!
_user_specified_name	input_1
��
�
__inference_call_7283

inputs:
'dense_20_matmul_readvariableop_resource:	<�7
(dense_20_biasadd_readvariableop_resource:	�B
3batch_normalization_16_cast_readvariableop_resource:	�D
5batch_normalization_16_cast_1_readvariableop_resource:	�;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�B
3batch_normalization_17_cast_readvariableop_resource:	�D
5batch_normalization_17_cast_1_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�B
3batch_normalization_18_cast_readvariableop_resource:	�D
5batch_normalization_18_cast_1_readvariableop_resource:	�;
'dense_23_matmul_readvariableop_resource:
��7
(dense_23_biasadd_readvariableop_resource:	�B
3batch_normalization_19_cast_readvariableop_resource:	�D
5batch_normalization_19_cast_1_readvariableop_resource:	�:
'dense_24_matmul_readvariableop_resource:	�<6
(dense_24_biasadd_readvariableop_resource:<:
'dense_25_matmul_readvariableop_resource:	�<6
(dense_25_biasadd_readvariableop_resource:<
identity

identity_1��*batch_normalization_16/Cast/ReadVariableOp�,batch_normalization_16/Cast_1/ReadVariableOp�*batch_normalization_17/Cast/ReadVariableOp�,batch_normalization_17/Cast_1/ReadVariableOp�*batch_normalization_18/Cast/ReadVariableOp�,batch_normalization_18/Cast_1/ReadVariableOp�*batch_normalization_19/Cast/ReadVariableOp�,batch_normalization_19/Cast_1/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	<�*
dtype0s
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Z
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0*
_output_shapes
:	�T
dense_20/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    t
dense_20/LessLessdense_20/BiasAdd:output:0dense_20/Less/y:output:0*
T0*
_output_shapes
:	�S
dense_20/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>i
dense_20/MulMuldense_20/Tanh:y:0dense_20/Mul/y:output:0*
T0*
_output_shapes
:	�
dense_20/SelectV2SelectV2dense_20/Less:z:0dense_20/Mul:z:0dense_20/Tanh:y:0*
T0*
_output_shapes
:	��
*batch_normalization_16/Cast/ReadVariableOpReadVariableOp3batch_normalization_16_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_16/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_16/batchnorm/addAddV24batch_normalization_16/Cast_1/ReadVariableOp:value:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_16/batchnorm/mulMuldense_20/SelectV2:output:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:	��
$batch_normalization_16/batchnorm/NegNeg2batch_normalization_16/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/mul_1Mul(batch_normalization_16/batchnorm/Neg:y:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/add_1AddV2(batch_normalization_16/batchnorm/mul:z:0*batch_normalization_16/batchnorm/mul_1:z:0*
T0*
_output_shapes
:	��
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMul*batch_normalization_16/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Z
dense_21/TanhTanhdense_21/BiasAdd:output:0*
T0*
_output_shapes
:	�T
dense_21/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    t
dense_21/LessLessdense_21/BiasAdd:output:0dense_21/Less/y:output:0*
T0*
_output_shapes
:	�S
dense_21/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>i
dense_21/MulMuldense_21/Tanh:y:0dense_21/Mul/y:output:0*
T0*
_output_shapes
:	�
dense_21/SelectV2SelectV2dense_21/Less:z:0dense_21/Mul:z:0dense_21/Tanh:y:0*
T0*
_output_shapes
:	��
*batch_normalization_17/Cast/ReadVariableOpReadVariableOp3batch_normalization_17_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_17/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_17_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_17/batchnorm/addAddV24batch_normalization_17/Cast_1/ReadVariableOp:value:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_17/batchnorm/mulMuldense_21/SelectV2:output:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:	��
$batch_normalization_17/batchnorm/NegNeg2batch_normalization_17/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/mul_1Mul(batch_normalization_17/batchnorm/Neg:y:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/add_1AddV2(batch_normalization_17/batchnorm/mul:z:0*batch_normalization_17/batchnorm/mul_1:z:0*
T0*
_output_shapes
:	��
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Z
dense_22/TanhTanhdense_22/BiasAdd:output:0*
T0*
_output_shapes
:	�T
dense_22/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    t
dense_22/LessLessdense_22/BiasAdd:output:0dense_22/Less/y:output:0*
T0*
_output_shapes
:	�S
dense_22/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>i
dense_22/MulMuldense_22/Tanh:y:0dense_22/Mul/y:output:0*
T0*
_output_shapes
:	�
dense_22/SelectV2SelectV2dense_22/Less:z:0dense_22/Mul:z:0dense_22/Tanh:y:0*
T0*
_output_shapes
:	��
*batch_normalization_18/Cast/ReadVariableOpReadVariableOp3batch_normalization_18_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_18/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_18_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_18/batchnorm/addAddV24batch_normalization_18/Cast_1/ReadVariableOp:value:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_18/batchnorm/mulMuldense_22/SelectV2:output:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:	��
$batch_normalization_18/batchnorm/NegNeg2batch_normalization_18/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/mul_1Mul(batch_normalization_18/batchnorm/Neg:y:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/add_1AddV2(batch_normalization_18/batchnorm/mul:z:0*batch_normalization_18/batchnorm/mul_1:z:0*
T0*
_output_shapes
:	��
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_23/MatMulMatMul*batch_normalization_18/batchnorm/add_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Z
dense_23/TanhTanhdense_23/BiasAdd:output:0*
T0*
_output_shapes
:	�T
dense_23/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    t
dense_23/LessLessdense_23/BiasAdd:output:0dense_23/Less/y:output:0*
T0*
_output_shapes
:	�S
dense_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>i
dense_23/MulMuldense_23/Tanh:y:0dense_23/Mul/y:output:0*
T0*
_output_shapes
:	�
dense_23/SelectV2SelectV2dense_23/Less:z:0dense_23/Mul:z:0dense_23/Tanh:y:0*
T0*
_output_shapes
:	��
*batch_normalization_19/Cast/ReadVariableOpReadVariableOp3batch_normalization_19_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_19/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_19_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_19/batchnorm/addAddV24batch_normalization_19/Cast_1/ReadVariableOp:value:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_19/batchnorm/mulMuldense_23/SelectV2:output:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:	��
$batch_normalization_19/batchnorm/NegNeg2batch_normalization_19/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/mul_1Mul(batch_normalization_19/batchnorm/Neg:y:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/add_1AddV2(batch_normalization_19/batchnorm/mul:z:0*batch_normalization_19/batchnorm/mul_1:z:0*
T0*
_output_shapes
:	��
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_24/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:<�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:<�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_25/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:<�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:<\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
clip_by_value/MinimumMinimumdense_25/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:<T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:<_
IdentityIdentitydense_24/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:<Y

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:<�
NoOpNoOp+^batch_normalization_16/Cast/ReadVariableOp-^batch_normalization_16/Cast_1/ReadVariableOp+^batch_normalization_17/Cast/ReadVariableOp-^batch_normalization_17/Cast_1/ReadVariableOp+^batch_normalization_18/Cast/ReadVariableOp-^batch_normalization_18/Cast_1/ReadVariableOp+^batch_normalization_19/Cast/ReadVariableOp-^batch_normalization_19/Cast_1/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:<: : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_16/Cast/ReadVariableOp*batch_normalization_16/Cast/ReadVariableOp2\
,batch_normalization_16/Cast_1/ReadVariableOp,batch_normalization_16/Cast_1/ReadVariableOp2X
*batch_normalization_17/Cast/ReadVariableOp*batch_normalization_17/Cast/ReadVariableOp2\
,batch_normalization_17/Cast_1/ReadVariableOp,batch_normalization_17/Cast_1/ReadVariableOp2X
*batch_normalization_18/Cast/ReadVariableOp*batch_normalization_18/Cast/ReadVariableOp2\
,batch_normalization_18/Cast_1/ReadVariableOp,batch_normalization_18/Cast_1/ReadVariableOp2X
*batch_normalization_19/Cast/ReadVariableOp*batch_normalization_19/Cast/ReadVariableOp2\
,batch_normalization_19/Cast_1/ReadVariableOp,batch_normalization_19/Cast_1/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:F B

_output_shapes

:<
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_15807142

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�d
batchnorm/mulMulinputsbatchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:����������T
batchnorm/NegNegmoments/Squeeze:output:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulbatchnorm/Neg:y:0batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul:z:0batchnorm/mul_1:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_15807906
input_1
unknown:	<�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�<

unknown_16:<

unknown_17:	�<

unknown_18:<
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������<:���������<*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__wrapped_model_15807021o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������<: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������<
!
_user_specified_name	input_1
�	
�
F__inference_dense_25_layer_call_and_return_conditional_losses_15807418

inputs1
matmul_readvariableop_resource:	�<-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������<w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_dense_23_layer_call_and_return_conditional_losses_15808420

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
LessLessBiasAdd:output:0Less/y:output:0*
T0*(
_output_shapes
:����������J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>W
MulMulTanh:y:0Mul/y:output:0*
T0*(
_output_shapes
:����������d
SelectV2SelectV2Less:z:0Mul:z:0Tanh:y:0*
T0*(
_output_shapes
:����������a
IdentityIdentitySelectV2:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
#__inference__wrapped_model_15807021
input_1/
policy_network_soft_15806977:	<�+
policy_network_soft_15806979:	�+
policy_network_soft_15806981:	�+
policy_network_soft_15806983:	�0
policy_network_soft_15806985:
��+
policy_network_soft_15806987:	�+
policy_network_soft_15806989:	�+
policy_network_soft_15806991:	�0
policy_network_soft_15806993:
��+
policy_network_soft_15806995:	�+
policy_network_soft_15806997:	�+
policy_network_soft_15806999:	�0
policy_network_soft_15807001:
��+
policy_network_soft_15807003:	�+
policy_network_soft_15807005:	�+
policy_network_soft_15807007:	�/
policy_network_soft_15807009:	�<*
policy_network_soft_15807011:</
policy_network_soft_15807013:	�<*
policy_network_soft_15807015:<
identity

identity_1��+policy_network_soft/StatefulPartitionedCall�
+policy_network_soft/StatefulPartitionedCallStatefulPartitionedCallinput_1policy_network_soft_15806977policy_network_soft_15806979policy_network_soft_15806981policy_network_soft_15806983policy_network_soft_15806985policy_network_soft_15806987policy_network_soft_15806989policy_network_soft_15806991policy_network_soft_15806993policy_network_soft_15806995policy_network_soft_15806997policy_network_soft_15806999policy_network_soft_15807001policy_network_soft_15807003policy_network_soft_15807005policy_network_soft_15807007policy_network_soft_15807009policy_network_soft_15807011policy_network_soft_15807013policy_network_soft_15807015* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������<:���������<*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_call_6289�
IdentityIdentity4policy_network_soft/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<�

Identity_1Identity4policy_network_soft/StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������<t
NoOpNoOp,^policy_network_soft/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������<: : : : : : : : : : : : : : : : : : : : 2Z
+policy_network_soft/StatefulPartitionedCall+policy_network_soft/StatefulPartitionedCall:P L
'
_output_shapes
:���������<
!
_user_specified_name	input_1
�
�
+__inference_dense_21_layer_call_fn_15808354

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_21_layer_call_and_return_conditional_losses_15807327p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_24_layer_call_fn_15808291

inputs
unknown:	�<
	unknown_0:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_15807402o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
__inference_call_27032

inputs:
'dense_20_matmul_readvariableop_resource:	<�7
(dense_20_biasadd_readvariableop_resource:	�M
>batch_normalization_16_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_16_assignmovingavg_1_readvariableop_resource:	�;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�M
>batch_normalization_17_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_17_assignmovingavg_1_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�M
>batch_normalization_18_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_18_assignmovingavg_1_readvariableop_resource:	�;
'dense_23_matmul_readvariableop_resource:
��7
(dense_23_biasadd_readvariableop_resource:	�M
>batch_normalization_19_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_19_assignmovingavg_1_readvariableop_resource:	�:
'dense_24_matmul_readvariableop_resource:	�<6
(dense_24_biasadd_readvariableop_resource:<:
'dense_25_matmul_readvariableop_resource:	�<6
(dense_25_biasadd_readvariableop_resource:<
identity

identity_1��&batch_normalization_16/AssignMovingAvg�5batch_normalization_16/AssignMovingAvg/ReadVariableOp�(batch_normalization_16/AssignMovingAvg_1�7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp�&batch_normalization_17/AssignMovingAvg�5batch_normalization_17/AssignMovingAvg/ReadVariableOp�(batch_normalization_17/AssignMovingAvg_1�7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp�&batch_normalization_18/AssignMovingAvg�5batch_normalization_18/AssignMovingAvg/ReadVariableOp�(batch_normalization_18/AssignMovingAvg_1�7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp�&batch_normalization_19/AssignMovingAvg�5batch_normalization_19/AssignMovingAvg/ReadVariableOp�(batch_normalization_19/AssignMovingAvg_1�7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	<�*
dtype0t
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
���
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��[
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0* 
_output_shapes
:
��T
dense_20/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    u
dense_20/LessLessdense_20/BiasAdd:output:0dense_20/Less/y:output:0*
T0* 
_output_shapes
:
��S
dense_20/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>j
dense_20/MulMuldense_20/Tanh:y:0dense_20/Mul/y:output:0*
T0* 
_output_shapes
:
���
dense_20/SelectV2SelectV2dense_20/Less:z:0dense_20/Mul:z:0dense_20/Tanh:y:0*
T0* 
_output_shapes
:
��
5batch_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_16/moments/meanMeandense_20/SelectV2:output:0>batch_normalization_16/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_16/moments/StopGradientStopGradient,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_16/moments/SquaredDifferenceSquaredDifferencedense_20/SelectV2:output:04batch_normalization_16/moments/StopGradient:output:0*
T0* 
_output_shapes
:
���
9batch_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_16/moments/varianceMean4batch_normalization_16/moments/SquaredDifference:z:0Bbatch_normalization_16/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_16/moments/SqueezeSqueeze,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_16/moments/Squeeze_1Squeeze0batch_normalization_16/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_16/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_16_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_16/AssignMovingAvg/subSub=batch_normalization_16/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_16/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_16/AssignMovingAvg/mulMul.batch_normalization_16/AssignMovingAvg/sub:z:05batch_normalization_16/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_16/AssignMovingAvgAssignSubVariableOp>batch_normalization_16_assignmovingavg_readvariableop_resource.batch_normalization_16/AssignMovingAvg/mul:z:06^batch_normalization_16/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_16/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_16_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_16/AssignMovingAvg_1/subSub?batch_normalization_16/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_16/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_16/AssignMovingAvg_1/mulMul0batch_normalization_16/AssignMovingAvg_1/sub:z:07batch_normalization_16/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_16/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_16_assignmovingavg_1_readvariableop_resource0batch_normalization_16/AssignMovingAvg_1/mul:z:08^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_16/batchnorm/addAddV21batch_normalization_16/moments/Squeeze_1:output:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_16/batchnorm/mulMuldense_20/SelectV2:output:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0* 
_output_shapes
:
���
$batch_normalization_16/batchnorm/NegNeg/batch_normalization_16/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/mul_1Mul(batch_normalization_16/batchnorm/Neg:y:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/add_1AddV2(batch_normalization_16/batchnorm/mul:z:0*batch_normalization_16/batchnorm/mul_1:z:0*
T0* 
_output_shapes
:
���
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMul*batch_normalization_16/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
���
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��[
dense_21/TanhTanhdense_21/BiasAdd:output:0*
T0* 
_output_shapes
:
��T
dense_21/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    u
dense_21/LessLessdense_21/BiasAdd:output:0dense_21/Less/y:output:0*
T0* 
_output_shapes
:
��S
dense_21/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>j
dense_21/MulMuldense_21/Tanh:y:0dense_21/Mul/y:output:0*
T0* 
_output_shapes
:
���
dense_21/SelectV2SelectV2dense_21/Less:z:0dense_21/Mul:z:0dense_21/Tanh:y:0*
T0* 
_output_shapes
:
��
5batch_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_17/moments/meanMeandense_21/SelectV2:output:0>batch_normalization_17/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_17/moments/StopGradientStopGradient,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_17/moments/SquaredDifferenceSquaredDifferencedense_21/SelectV2:output:04batch_normalization_17/moments/StopGradient:output:0*
T0* 
_output_shapes
:
���
9batch_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_17/moments/varianceMean4batch_normalization_17/moments/SquaredDifference:z:0Bbatch_normalization_17/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_17/moments/SqueezeSqueeze,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_17/moments/Squeeze_1Squeeze0batch_normalization_17/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_17/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_17/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_17_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_17/AssignMovingAvg/subSub=batch_normalization_17/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_17/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_17/AssignMovingAvg/mulMul.batch_normalization_17/AssignMovingAvg/sub:z:05batch_normalization_17/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_17/AssignMovingAvgAssignSubVariableOp>batch_normalization_17_assignmovingavg_readvariableop_resource.batch_normalization_17/AssignMovingAvg/mul:z:06^batch_normalization_17/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_17/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_17_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_17/AssignMovingAvg_1/subSub?batch_normalization_17/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_17/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_17/AssignMovingAvg_1/mulMul0batch_normalization_17/AssignMovingAvg_1/sub:z:07batch_normalization_17/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_17/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_17_assignmovingavg_1_readvariableop_resource0batch_normalization_17/AssignMovingAvg_1/mul:z:08^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_17/batchnorm/addAddV21batch_normalization_17/moments/Squeeze_1:output:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_17/batchnorm/mulMuldense_21/SelectV2:output:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0* 
_output_shapes
:
���
$batch_normalization_17/batchnorm/NegNeg/batch_normalization_17/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/mul_1Mul(batch_normalization_17/batchnorm/Neg:y:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/add_1AddV2(batch_normalization_17/batchnorm/mul:z:0*batch_normalization_17/batchnorm/mul_1:z:0*
T0* 
_output_shapes
:
���
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
���
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��[
dense_22/TanhTanhdense_22/BiasAdd:output:0*
T0* 
_output_shapes
:
��T
dense_22/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    u
dense_22/LessLessdense_22/BiasAdd:output:0dense_22/Less/y:output:0*
T0* 
_output_shapes
:
��S
dense_22/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>j
dense_22/MulMuldense_22/Tanh:y:0dense_22/Mul/y:output:0*
T0* 
_output_shapes
:
���
dense_22/SelectV2SelectV2dense_22/Less:z:0dense_22/Mul:z:0dense_22/Tanh:y:0*
T0* 
_output_shapes
:
��
5batch_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_18/moments/meanMeandense_22/SelectV2:output:0>batch_normalization_18/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_18/moments/StopGradientStopGradient,batch_normalization_18/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_18/moments/SquaredDifferenceSquaredDifferencedense_22/SelectV2:output:04batch_normalization_18/moments/StopGradient:output:0*
T0* 
_output_shapes
:
���
9batch_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_18/moments/varianceMean4batch_normalization_18/moments/SquaredDifference:z:0Bbatch_normalization_18/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_18/moments/SqueezeSqueeze,batch_normalization_18/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_18/moments/Squeeze_1Squeeze0batch_normalization_18/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_18/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_18/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_18_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_18/AssignMovingAvg/subSub=batch_normalization_18/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_18/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_18/AssignMovingAvg/mulMul.batch_normalization_18/AssignMovingAvg/sub:z:05batch_normalization_18/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_18/AssignMovingAvgAssignSubVariableOp>batch_normalization_18_assignmovingavg_readvariableop_resource.batch_normalization_18/AssignMovingAvg/mul:z:06^batch_normalization_18/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_18/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_18_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_18/AssignMovingAvg_1/subSub?batch_normalization_18/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_18/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_18/AssignMovingAvg_1/mulMul0batch_normalization_18/AssignMovingAvg_1/sub:z:07batch_normalization_18/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_18/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_18_assignmovingavg_1_readvariableop_resource0batch_normalization_18/AssignMovingAvg_1/mul:z:08^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_18/batchnorm/addAddV21batch_normalization_18/moments/Squeeze_1:output:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_18/batchnorm/mulMuldense_22/SelectV2:output:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0* 
_output_shapes
:
���
$batch_normalization_18/batchnorm/NegNeg/batch_normalization_18/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/mul_1Mul(batch_normalization_18/batchnorm/Neg:y:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/add_1AddV2(batch_normalization_18/batchnorm/mul:z:0*batch_normalization_18/batchnorm/mul_1:z:0*
T0* 
_output_shapes
:
���
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_23/MatMulMatMul*batch_normalization_18/batchnorm/add_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
���
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��[
dense_23/TanhTanhdense_23/BiasAdd:output:0*
T0* 
_output_shapes
:
��T
dense_23/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    u
dense_23/LessLessdense_23/BiasAdd:output:0dense_23/Less/y:output:0*
T0* 
_output_shapes
:
��S
dense_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>j
dense_23/MulMuldense_23/Tanh:y:0dense_23/Mul/y:output:0*
T0* 
_output_shapes
:
���
dense_23/SelectV2SelectV2dense_23/Less:z:0dense_23/Mul:z:0dense_23/Tanh:y:0*
T0* 
_output_shapes
:
��
5batch_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_19/moments/meanMeandense_23/SelectV2:output:0>batch_normalization_19/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_19/moments/StopGradientStopGradient,batch_normalization_19/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_19/moments/SquaredDifferenceSquaredDifferencedense_23/SelectV2:output:04batch_normalization_19/moments/StopGradient:output:0*
T0* 
_output_shapes
:
���
9batch_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_19/moments/varianceMean4batch_normalization_19/moments/SquaredDifference:z:0Bbatch_normalization_19/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_19/moments/SqueezeSqueeze,batch_normalization_19/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_19/moments/Squeeze_1Squeeze0batch_normalization_19/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_19/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_19/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_19_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_19/AssignMovingAvg/subSub=batch_normalization_19/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_19/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_19/AssignMovingAvg/mulMul.batch_normalization_19/AssignMovingAvg/sub:z:05batch_normalization_19/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_19/AssignMovingAvgAssignSubVariableOp>batch_normalization_19_assignmovingavg_readvariableop_resource.batch_normalization_19/AssignMovingAvg/mul:z:06^batch_normalization_19/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_19/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_19_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_19/AssignMovingAvg_1/subSub?batch_normalization_19/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_19/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_19/AssignMovingAvg_1/mulMul0batch_normalization_19/AssignMovingAvg_1/sub:z:07batch_normalization_19/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_19/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_19_assignmovingavg_1_readvariableop_resource0batch_normalization_19/AssignMovingAvg_1/mul:z:08^batch_normalization_19/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_19/batchnorm/addAddV21batch_normalization_19/moments/Squeeze_1:output:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_19/batchnorm/mulMuldense_23/SelectV2:output:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0* 
_output_shapes
:
���
$batch_normalization_19/batchnorm/NegNeg/batch_normalization_19/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/mul_1Mul(batch_normalization_19/batchnorm/Neg:y:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/add_1AddV2(batch_normalization_19/batchnorm/mul:z:0*batch_normalization_19/batchnorm/mul_1:z:0*
T0* 
_output_shapes
:
���
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_24/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�<�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�<�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_25/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�<�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�<\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
clip_by_value/MinimumMinimumdense_25/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:	�<T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��w
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:	�<`
IdentityIdentitydense_24/BiasAdd:output:0^NoOp*
T0*
_output_shapes
:	�<Z

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes
:	�<�	
NoOpNoOp'^batch_normalization_16/AssignMovingAvg6^batch_normalization_16/AssignMovingAvg/ReadVariableOp)^batch_normalization_16/AssignMovingAvg_18^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp'^batch_normalization_17/AssignMovingAvg6^batch_normalization_17/AssignMovingAvg/ReadVariableOp)^batch_normalization_17/AssignMovingAvg_18^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp'^batch_normalization_18/AssignMovingAvg6^batch_normalization_18/AssignMovingAvg/ReadVariableOp)^batch_normalization_18/AssignMovingAvg_18^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp'^batch_normalization_19/AssignMovingAvg6^batch_normalization_19/AssignMovingAvg/ReadVariableOp)^batch_normalization_19/AssignMovingAvg_18^batch_normalization_19/AssignMovingAvg_1/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:	�<: : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_16/AssignMovingAvg&batch_normalization_16/AssignMovingAvg2n
5batch_normalization_16/AssignMovingAvg/ReadVariableOp5batch_normalization_16/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_16/AssignMovingAvg_1(batch_normalization_16/AssignMovingAvg_12r
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp2P
&batch_normalization_17/AssignMovingAvg&batch_normalization_17/AssignMovingAvg2n
5batch_normalization_17/AssignMovingAvg/ReadVariableOp5batch_normalization_17/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_17/AssignMovingAvg_1(batch_normalization_17/AssignMovingAvg_12r
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp2P
&batch_normalization_18/AssignMovingAvg&batch_normalization_18/AssignMovingAvg2n
5batch_normalization_18/AssignMovingAvg/ReadVariableOp5batch_normalization_18/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_18/AssignMovingAvg_1(batch_normalization_18/AssignMovingAvg_12r
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp2P
&batch_normalization_19/AssignMovingAvg&batch_normalization_19/AssignMovingAvg2n
5batch_normalization_19/AssignMovingAvg/ReadVariableOp5batch_normalization_19/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_19/AssignMovingAvg_1(batch_normalization_19/AssignMovingAvg_12r
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:G C

_output_shapes
:	�<
 
_user_specified_nameinputs
�
�
6__inference_policy_network_soft_layer_call_fn_15808000

inputs
unknown:	<�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�<

unknown_16:<

unknown_17:	�<

unknown_18:<
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������<:���������<*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15807643o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������<: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_15808515

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�d
batchnorm/mulMulinputsbatchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:����������W
batchnorm/NegNegCast/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulbatchnorm/Neg:y:0batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul:z:0batchnorm/mul_1:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������t
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�t
�
!__inference__traced_save_15808872
file_prefix.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableopA
=savev2_batch_normalization_18_moving_mean_read_readvariableopE
Asavev2_batch_normalization_18_moving_variance_read_readvariableopA
=savev2_batch_normalization_19_moving_mean_read_readvariableopE
Asavev2_batch_normalization_19_moving_variance_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop#
savev2_iter_read_readvariableop	7
3savev2_adam_2_dense_20_kernel_m_read_readvariableop5
1savev2_adam_2_dense_20_bias_m_read_readvariableop7
3savev2_adam_2_dense_21_kernel_m_read_readvariableop5
1savev2_adam_2_dense_21_bias_m_read_readvariableop7
3savev2_adam_2_dense_22_kernel_m_read_readvariableop5
1savev2_adam_2_dense_22_bias_m_read_readvariableop7
3savev2_adam_2_dense_23_kernel_m_read_readvariableop5
1savev2_adam_2_dense_23_bias_m_read_readvariableop7
3savev2_adam_2_dense_24_kernel_m_read_readvariableop5
1savev2_adam_2_dense_24_bias_m_read_readvariableop7
3savev2_adam_2_dense_25_kernel_m_read_readvariableop5
1savev2_adam_2_dense_25_bias_m_read_readvariableop7
3savev2_adam_2_dense_20_kernel_v_read_readvariableop5
1savev2_adam_2_dense_20_bias_v_read_readvariableop7
3savev2_adam_2_dense_21_kernel_v_read_readvariableop5
1savev2_adam_2_dense_21_bias_v_read_readvariableop7
3savev2_adam_2_dense_22_kernel_v_read_readvariableop5
1savev2_adam_2_dense_22_bias_v_read_readvariableop7
3savev2_adam_2_dense_23_kernel_v_read_readvariableop5
1savev2_adam_2_dense_23_bias_v_read_readvariableop7
3savev2_adam_2_dense_24_kernel_v_read_readvariableop5
1savev2_adam_2_dense_24_bias_v_read_readvariableop7
3savev2_adam_2_dense_25_kernel_v_read_readvariableop5
1savev2_adam_2_dense_25_bias_v_read_readvariableop:
6savev2_adam_2_dense_20_kernel_vhat_read_readvariableop8
4savev2_adam_2_dense_20_bias_vhat_read_readvariableop:
6savev2_adam_2_dense_21_kernel_vhat_read_readvariableop8
4savev2_adam_2_dense_21_bias_vhat_read_readvariableop:
6savev2_adam_2_dense_22_kernel_vhat_read_readvariableop8
4savev2_adam_2_dense_22_bias_vhat_read_readvariableop:
6savev2_adam_2_dense_23_kernel_vhat_read_readvariableop8
4savev2_adam_2_dense_23_bias_vhat_read_readvariableop:
6savev2_adam_2_dense_24_kernel_vhat_read_readvariableop8
4savev2_adam_2_dense_24_bias_vhat_read_readvariableop:
6savev2_adam_2_dense_25_kernel_vhat_read_readvariableop8
4savev2_adam_2_dense_25_bias_vhat_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/6/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/7/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBFvariables/16/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBFvariables/17/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBFvariables/18/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBFvariables/19/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableop=savev2_batch_normalization_18_moving_mean_read_readvariableopAsavev2_batch_normalization_18_moving_variance_read_readvariableop=savev2_batch_normalization_19_moving_mean_read_readvariableopAsavev2_batch_normalization_19_moving_variance_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableopsavev2_iter_read_readvariableop3savev2_adam_2_dense_20_kernel_m_read_readvariableop1savev2_adam_2_dense_20_bias_m_read_readvariableop3savev2_adam_2_dense_21_kernel_m_read_readvariableop1savev2_adam_2_dense_21_bias_m_read_readvariableop3savev2_adam_2_dense_22_kernel_m_read_readvariableop1savev2_adam_2_dense_22_bias_m_read_readvariableop3savev2_adam_2_dense_23_kernel_m_read_readvariableop1savev2_adam_2_dense_23_bias_m_read_readvariableop3savev2_adam_2_dense_24_kernel_m_read_readvariableop1savev2_adam_2_dense_24_bias_m_read_readvariableop3savev2_adam_2_dense_25_kernel_m_read_readvariableop1savev2_adam_2_dense_25_bias_m_read_readvariableop3savev2_adam_2_dense_20_kernel_v_read_readvariableop1savev2_adam_2_dense_20_bias_v_read_readvariableop3savev2_adam_2_dense_21_kernel_v_read_readvariableop1savev2_adam_2_dense_21_bias_v_read_readvariableop3savev2_adam_2_dense_22_kernel_v_read_readvariableop1savev2_adam_2_dense_22_bias_v_read_readvariableop3savev2_adam_2_dense_23_kernel_v_read_readvariableop1savev2_adam_2_dense_23_bias_v_read_readvariableop3savev2_adam_2_dense_24_kernel_v_read_readvariableop1savev2_adam_2_dense_24_bias_v_read_readvariableop3savev2_adam_2_dense_25_kernel_v_read_readvariableop1savev2_adam_2_dense_25_bias_v_read_readvariableop6savev2_adam_2_dense_20_kernel_vhat_read_readvariableop4savev2_adam_2_dense_20_bias_vhat_read_readvariableop6savev2_adam_2_dense_21_kernel_vhat_read_readvariableop4savev2_adam_2_dense_21_bias_vhat_read_readvariableop6savev2_adam_2_dense_22_kernel_vhat_read_readvariableop4savev2_adam_2_dense_22_bias_vhat_read_readvariableop6savev2_adam_2_dense_23_kernel_vhat_read_readvariableop4savev2_adam_2_dense_23_bias_vhat_read_readvariableop6savev2_adam_2_dense_24_kernel_vhat_read_readvariableop4savev2_adam_2_dense_24_bias_vhat_read_readvariableop6savev2_adam_2_dense_25_kernel_vhat_read_readvariableop4savev2_adam_2_dense_25_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *K
dtypesA
?2=	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :	<�:�:
��:�:
��:�:
��:�:�:�:�:�:�:�:�:�:	�<:<:	�<:<: : : : :	<�:�:
��:�:
��:�:
��:�:	�<:<:	�<:<:	<�:�:
��:�:
��:�:
��:�:	�<:<:	�<:<:	<�:�:
��:�:
��:�:
��:�:	�<:<:	�<:<: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	<�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:!	

_output_shapes	
:�:!


_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�<: 

_output_shapes
:<:%!

_output_shapes
:	�<: 

_output_shapes
:<:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	<�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:! 

_output_shapes	
:�:%!!

_output_shapes
:	�<: "

_output_shapes
:<:%#!

_output_shapes
:	�<: $

_output_shapes
:<:%%!

_output_shapes
:	<�:!&

_output_shapes	
:�:&'"
 
_output_shapes
:
��:!(

_output_shapes	
:�:&)"
 
_output_shapes
:
��:!*

_output_shapes	
:�:&+"
 
_output_shapes
:
��:!,

_output_shapes	
:�:%-!

_output_shapes
:	�<: .

_output_shapes
:<:%/!

_output_shapes
:	�<: 0

_output_shapes
:<:%1!

_output_shapes
:	<�:!2

_output_shapes	
:�:&3"
 
_output_shapes
:
��:!4

_output_shapes	
:�:&5"
 
_output_shapes
:
��:!6

_output_shapes	
:�:&7"
 
_output_shapes
:
��:!8

_output_shapes	
:�:%9!

_output_shapes
:	�<: :

_output_shapes
:<:%;!

_output_shapes
:	�<: <

_output_shapes
:<:=

_output_shapes
: 
�
�
9__inference_batch_normalization_17_layer_call_fn_15808491

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_15807104p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_dense_21_layer_call_and_return_conditional_losses_15807327

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
LessLessBiasAdd:output:0Less/y:output:0*
T0*(
_output_shapes
:����������J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>W
MulMulTanh:y:0Mul/y:output:0*
T0*(
_output_shapes
:����������d
SelectV2SelectV2Less:z:0Mul:z:0Tanh:y:0*
T0*(
_output_shapes
:����������a
IdentityIdentitySelectV2:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_19_layer_call_fn_15808624

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_19_layer_call_and_return_conditional_losses_15807270p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15808282

inputs:
'dense_20_matmul_readvariableop_resource:	<�7
(dense_20_biasadd_readvariableop_resource:	�M
>batch_normalization_16_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_16_assignmovingavg_1_readvariableop_resource:	�;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�M
>batch_normalization_17_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_17_assignmovingavg_1_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�M
>batch_normalization_18_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_18_assignmovingavg_1_readvariableop_resource:	�;
'dense_23_matmul_readvariableop_resource:
��7
(dense_23_biasadd_readvariableop_resource:	�M
>batch_normalization_19_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_19_assignmovingavg_1_readvariableop_resource:	�:
'dense_24_matmul_readvariableop_resource:	�<6
(dense_24_biasadd_readvariableop_resource:<:
'dense_25_matmul_readvariableop_resource:	�<6
(dense_25_biasadd_readvariableop_resource:<
identity

identity_1��&batch_normalization_16/AssignMovingAvg�5batch_normalization_16/AssignMovingAvg/ReadVariableOp�(batch_normalization_16/AssignMovingAvg_1�7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp�&batch_normalization_17/AssignMovingAvg�5batch_normalization_17/AssignMovingAvg/ReadVariableOp�(batch_normalization_17/AssignMovingAvg_1�7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp�&batch_normalization_18/AssignMovingAvg�5batch_normalization_18/AssignMovingAvg/ReadVariableOp�(batch_normalization_18/AssignMovingAvg_1�7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp�&batch_normalization_19/AssignMovingAvg�5batch_normalization_19/AssignMovingAvg/ReadVariableOp�(batch_normalization_19/AssignMovingAvg_1�7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	<�*
dtype0|
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0*(
_output_shapes
:����������T
dense_20/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
dense_20/LessLessdense_20/BiasAdd:output:0dense_20/Less/y:output:0*
T0*(
_output_shapes
:����������S
dense_20/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>r
dense_20/MulMuldense_20/Tanh:y:0dense_20/Mul/y:output:0*
T0*(
_output_shapes
:�����������
dense_20/SelectV2SelectV2dense_20/Less:z:0dense_20/Mul:z:0dense_20/Tanh:y:0*
T0*(
_output_shapes
:����������
5batch_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_16/moments/meanMeandense_20/SelectV2:output:0>batch_normalization_16/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_16/moments/StopGradientStopGradient,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_16/moments/SquaredDifferenceSquaredDifferencedense_20/SelectV2:output:04batch_normalization_16/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_16/moments/varianceMean4batch_normalization_16/moments/SquaredDifference:z:0Bbatch_normalization_16/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_16/moments/SqueezeSqueeze,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_16/moments/Squeeze_1Squeeze0batch_normalization_16/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_16/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_16_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_16/AssignMovingAvg/subSub=batch_normalization_16/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_16/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_16/AssignMovingAvg/mulMul.batch_normalization_16/AssignMovingAvg/sub:z:05batch_normalization_16/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_16/AssignMovingAvgAssignSubVariableOp>batch_normalization_16_assignmovingavg_readvariableop_resource.batch_normalization_16/AssignMovingAvg/mul:z:06^batch_normalization_16/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_16/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_16_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_16/AssignMovingAvg_1/subSub?batch_normalization_16/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_16/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_16/AssignMovingAvg_1/mulMul0batch_normalization_16/AssignMovingAvg_1/sub:z:07batch_normalization_16/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_16/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_16_assignmovingavg_1_readvariableop_resource0batch_normalization_16/AssignMovingAvg_1/mul:z:08^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_16/batchnorm/addAddV21batch_normalization_16/moments/Squeeze_1:output:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_16/batchnorm/mulMuldense_20/SelectV2:output:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
$batch_normalization_16/batchnorm/NegNeg/batch_normalization_16/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/mul_1Mul(batch_normalization_16/batchnorm/Neg:y:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/add_1AddV2(batch_normalization_16/batchnorm/mul:z:0*batch_normalization_16/batchnorm/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMul*batch_normalization_16/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_21/TanhTanhdense_21/BiasAdd:output:0*
T0*(
_output_shapes
:����������T
dense_21/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
dense_21/LessLessdense_21/BiasAdd:output:0dense_21/Less/y:output:0*
T0*(
_output_shapes
:����������S
dense_21/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>r
dense_21/MulMuldense_21/Tanh:y:0dense_21/Mul/y:output:0*
T0*(
_output_shapes
:�����������
dense_21/SelectV2SelectV2dense_21/Less:z:0dense_21/Mul:z:0dense_21/Tanh:y:0*
T0*(
_output_shapes
:����������
5batch_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_17/moments/meanMeandense_21/SelectV2:output:0>batch_normalization_17/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_17/moments/StopGradientStopGradient,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_17/moments/SquaredDifferenceSquaredDifferencedense_21/SelectV2:output:04batch_normalization_17/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_17/moments/varianceMean4batch_normalization_17/moments/SquaredDifference:z:0Bbatch_normalization_17/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_17/moments/SqueezeSqueeze,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_17/moments/Squeeze_1Squeeze0batch_normalization_17/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_17/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_17/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_17_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_17/AssignMovingAvg/subSub=batch_normalization_17/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_17/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_17/AssignMovingAvg/mulMul.batch_normalization_17/AssignMovingAvg/sub:z:05batch_normalization_17/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_17/AssignMovingAvgAssignSubVariableOp>batch_normalization_17_assignmovingavg_readvariableop_resource.batch_normalization_17/AssignMovingAvg/mul:z:06^batch_normalization_17/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_17/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_17_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_17/AssignMovingAvg_1/subSub?batch_normalization_17/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_17/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_17/AssignMovingAvg_1/mulMul0batch_normalization_17/AssignMovingAvg_1/sub:z:07batch_normalization_17/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_17/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_17_assignmovingavg_1_readvariableop_resource0batch_normalization_17/AssignMovingAvg_1/mul:z:08^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_17/batchnorm/addAddV21batch_normalization_17/moments/Squeeze_1:output:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_17/batchnorm/mulMuldense_21/SelectV2:output:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
$batch_normalization_17/batchnorm/NegNeg/batch_normalization_17/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/mul_1Mul(batch_normalization_17/batchnorm/Neg:y:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/add_1AddV2(batch_normalization_17/batchnorm/mul:z:0*batch_normalization_17/batchnorm/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_22/TanhTanhdense_22/BiasAdd:output:0*
T0*(
_output_shapes
:����������T
dense_22/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
dense_22/LessLessdense_22/BiasAdd:output:0dense_22/Less/y:output:0*
T0*(
_output_shapes
:����������S
dense_22/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>r
dense_22/MulMuldense_22/Tanh:y:0dense_22/Mul/y:output:0*
T0*(
_output_shapes
:�����������
dense_22/SelectV2SelectV2dense_22/Less:z:0dense_22/Mul:z:0dense_22/Tanh:y:0*
T0*(
_output_shapes
:����������
5batch_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_18/moments/meanMeandense_22/SelectV2:output:0>batch_normalization_18/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_18/moments/StopGradientStopGradient,batch_normalization_18/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_18/moments/SquaredDifferenceSquaredDifferencedense_22/SelectV2:output:04batch_normalization_18/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_18/moments/varianceMean4batch_normalization_18/moments/SquaredDifference:z:0Bbatch_normalization_18/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_18/moments/SqueezeSqueeze,batch_normalization_18/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_18/moments/Squeeze_1Squeeze0batch_normalization_18/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_18/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_18/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_18_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_18/AssignMovingAvg/subSub=batch_normalization_18/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_18/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_18/AssignMovingAvg/mulMul.batch_normalization_18/AssignMovingAvg/sub:z:05batch_normalization_18/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_18/AssignMovingAvgAssignSubVariableOp>batch_normalization_18_assignmovingavg_readvariableop_resource.batch_normalization_18/AssignMovingAvg/mul:z:06^batch_normalization_18/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_18/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_18_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_18/AssignMovingAvg_1/subSub?batch_normalization_18/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_18/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_18/AssignMovingAvg_1/mulMul0batch_normalization_18/AssignMovingAvg_1/sub:z:07batch_normalization_18/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_18/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_18_assignmovingavg_1_readvariableop_resource0batch_normalization_18/AssignMovingAvg_1/mul:z:08^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_18/batchnorm/addAddV21batch_normalization_18/moments/Squeeze_1:output:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_18/batchnorm/mulMuldense_22/SelectV2:output:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
$batch_normalization_18/batchnorm/NegNeg/batch_normalization_18/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/mul_1Mul(batch_normalization_18/batchnorm/Neg:y:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/add_1AddV2(batch_normalization_18/batchnorm/mul:z:0*batch_normalization_18/batchnorm/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_23/MatMulMatMul*batch_normalization_18/batchnorm/add_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_23/TanhTanhdense_23/BiasAdd:output:0*
T0*(
_output_shapes
:����������T
dense_23/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
dense_23/LessLessdense_23/BiasAdd:output:0dense_23/Less/y:output:0*
T0*(
_output_shapes
:����������S
dense_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>r
dense_23/MulMuldense_23/Tanh:y:0dense_23/Mul/y:output:0*
T0*(
_output_shapes
:�����������
dense_23/SelectV2SelectV2dense_23/Less:z:0dense_23/Mul:z:0dense_23/Tanh:y:0*
T0*(
_output_shapes
:����������
5batch_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_19/moments/meanMeandense_23/SelectV2:output:0>batch_normalization_19/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_19/moments/StopGradientStopGradient,batch_normalization_19/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_19/moments/SquaredDifferenceSquaredDifferencedense_23/SelectV2:output:04batch_normalization_19/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_19/moments/varianceMean4batch_normalization_19/moments/SquaredDifference:z:0Bbatch_normalization_19/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_19/moments/SqueezeSqueeze,batch_normalization_19/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_19/moments/Squeeze_1Squeeze0batch_normalization_19/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_19/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_19/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_19_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_19/AssignMovingAvg/subSub=batch_normalization_19/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_19/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_19/AssignMovingAvg/mulMul.batch_normalization_19/AssignMovingAvg/sub:z:05batch_normalization_19/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_19/AssignMovingAvgAssignSubVariableOp>batch_normalization_19_assignmovingavg_readvariableop_resource.batch_normalization_19/AssignMovingAvg/mul:z:06^batch_normalization_19/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_19/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_19_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_19/AssignMovingAvg_1/subSub?batch_normalization_19/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_19/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_19/AssignMovingAvg_1/mulMul0batch_normalization_19/AssignMovingAvg_1/sub:z:07batch_normalization_19/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_19/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_19_assignmovingavg_1_readvariableop_resource0batch_normalization_19/AssignMovingAvg_1/mul:z:08^batch_normalization_19/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_19/batchnorm/addAddV21batch_normalization_19/moments/Squeeze_1:output:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_19/batchnorm/mulMuldense_23/SelectV2:output:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
$batch_normalization_19/batchnorm/NegNeg/batch_normalization_19/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/mul_1Mul(batch_normalization_19/batchnorm/Neg:y:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/add_1AddV2(batch_normalization_19/batchnorm/mul:z:0*batch_normalization_19/batchnorm/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_24/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_25/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
clip_by_value/MinimumMinimumdense_25/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������<T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������<h
IdentityIdentitydense_24/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������<b

Identity_1Identityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:���������<�	
NoOpNoOp'^batch_normalization_16/AssignMovingAvg6^batch_normalization_16/AssignMovingAvg/ReadVariableOp)^batch_normalization_16/AssignMovingAvg_18^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp'^batch_normalization_17/AssignMovingAvg6^batch_normalization_17/AssignMovingAvg/ReadVariableOp)^batch_normalization_17/AssignMovingAvg_18^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp'^batch_normalization_18/AssignMovingAvg6^batch_normalization_18/AssignMovingAvg/ReadVariableOp)^batch_normalization_18/AssignMovingAvg_18^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp'^batch_normalization_19/AssignMovingAvg6^batch_normalization_19/AssignMovingAvg/ReadVariableOp)^batch_normalization_19/AssignMovingAvg_18^batch_normalization_19/AssignMovingAvg_1/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������<: : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_16/AssignMovingAvg&batch_normalization_16/AssignMovingAvg2n
5batch_normalization_16/AssignMovingAvg/ReadVariableOp5batch_normalization_16/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_16/AssignMovingAvg_1(batch_normalization_16/AssignMovingAvg_12r
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp2P
&batch_normalization_17/AssignMovingAvg&batch_normalization_17/AssignMovingAvg2n
5batch_normalization_17/AssignMovingAvg/ReadVariableOp5batch_normalization_17/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_17/AssignMovingAvg_1(batch_normalization_17/AssignMovingAvg_12r
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp2P
&batch_normalization_18/AssignMovingAvg&batch_normalization_18/AssignMovingAvg2n
5batch_normalization_18/AssignMovingAvg/ReadVariableOp5batch_normalization_18/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_18/AssignMovingAvg_1(batch_normalization_18/AssignMovingAvg_12r
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp2P
&batch_normalization_19/AssignMovingAvg&batch_normalization_19/AssignMovingAvg2n
5batch_normalization_19/AssignMovingAvg/ReadVariableOp5batch_normalization_19/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_19/AssignMovingAvg_1(batch_normalization_19/AssignMovingAvg_12r
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_17_layer_call_fn_15808500

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_15807142p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_19_layer_call_and_return_conditional_losses_15808639

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�d
batchnorm/mulMulinputsbatchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:����������W
batchnorm/NegNegCast/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulbatchnorm/Neg:y:0batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul:z:0batchnorm/mul_1:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������t
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_dense_24_layer_call_and_return_conditional_losses_15807402

inputs1
matmul_readvariableop_resource:	�<-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������<w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�>
�

Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15807853
input_1$
dense_20_15807797:	<� 
dense_20_15807799:	�.
batch_normalization_16_15807802:	�.
batch_normalization_16_15807804:	�%
dense_21_15807807:
�� 
dense_21_15807809:	�.
batch_normalization_17_15807812:	�.
batch_normalization_17_15807814:	�%
dense_22_15807817:
�� 
dense_22_15807819:	�.
batch_normalization_18_15807822:	�.
batch_normalization_18_15807824:	�%
dense_23_15807827:
�� 
dense_23_15807829:	�.
batch_normalization_19_15807832:	�.
batch_normalization_19_15807834:	�$
dense_24_15807837:	�<
dense_24_15807839:<$
dense_25_15807842:	�<
dense_25_15807844:<
identity

identity_1��.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall�.batch_normalization_18/StatefulPartitionedCall�.batch_normalization_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall� dense_25/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_20_15807797dense_20_15807799*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_20_layer_call_and_return_conditional_losses_15807300�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_16_15807802batch_normalization_16_15807804*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_15807078�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_21_15807807dense_21_15807809*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_21_layer_call_and_return_conditional_losses_15807327�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_17_15807812batch_normalization_17_15807814*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_15807142�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_22_15807817dense_22_15807819*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_22_layer_call_and_return_conditional_losses_15807354�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_18_15807822batch_normalization_18_15807824*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_18_layer_call_and_return_conditional_losses_15807206�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0dense_23_15807827dense_23_15807829*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_15807381�
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0batch_normalization_19_15807832batch_normalization_19_15807834*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_19_layer_call_and_return_conditional_losses_15807270�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0dense_24_15807837dense_24_15807839*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_15807402�
 dense_25/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0dense_25_15807842dense_25_15807844*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_15807418\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
clip_by_value/MinimumMinimum)dense_25/StatefulPartitionedCall:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������<T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������<x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<b

Identity_1Identityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:���������<�
NoOpNoOp/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������<: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:P L
'
_output_shapes
:���������<
!
_user_specified_name	input_1
�
�
F__inference_dense_21_layer_call_and_return_conditional_losses_15808370

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
LessLessBiasAdd:output:0Less/y:output:0*
T0*(
_output_shapes
:����������J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>W
MulMulTanh:y:0Mul/y:output:0*
T0*(
_output_shapes
:����������d
SelectV2SelectV2Less:z:0Mul:z:0Tanh:y:0*
T0*(
_output_shapes
:����������a
IdentityIdentitySelectV2:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
݃
�
__inference_call_7742

inputs:
'dense_20_matmul_readvariableop_resource:	<�7
(dense_20_biasadd_readvariableop_resource:	�B
3batch_normalization_16_cast_readvariableop_resource:	�D
5batch_normalization_16_cast_1_readvariableop_resource:	�;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�B
3batch_normalization_17_cast_readvariableop_resource:	�D
5batch_normalization_17_cast_1_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�B
3batch_normalization_18_cast_readvariableop_resource:	�D
5batch_normalization_18_cast_1_readvariableop_resource:	�;
'dense_23_matmul_readvariableop_resource:
��7
(dense_23_biasadd_readvariableop_resource:	�B
3batch_normalization_19_cast_readvariableop_resource:	�D
5batch_normalization_19_cast_1_readvariableop_resource:	�:
'dense_24_matmul_readvariableop_resource:	�<6
(dense_24_biasadd_readvariableop_resource:<:
'dense_25_matmul_readvariableop_resource:	�<6
(dense_25_biasadd_readvariableop_resource:<
identity

identity_1��*batch_normalization_16/Cast/ReadVariableOp�,batch_normalization_16/Cast_1/ReadVariableOp�*batch_normalization_17/Cast/ReadVariableOp�,batch_normalization_17/Cast_1/ReadVariableOp�*batch_normalization_18/Cast/ReadVariableOp�,batch_normalization_18/Cast_1/ReadVariableOp�*batch_normalization_19/Cast/ReadVariableOp�,batch_normalization_19/Cast_1/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOpU
dense_20/CastCastinputs*

DstT0*

SrcT0*
_output_shapes

:<�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	<�*
dtype0~
dense_20/MatMulMatMuldense_20/Cast:y:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Z
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0*
_output_shapes
:	�T
dense_20/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    t
dense_20/LessLessdense_20/BiasAdd:output:0dense_20/Less/y:output:0*
T0*
_output_shapes
:	�S
dense_20/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>i
dense_20/MulMuldense_20/Tanh:y:0dense_20/Mul/y:output:0*
T0*
_output_shapes
:	�
dense_20/SelectV2SelectV2dense_20/Less:z:0dense_20/Mul:z:0dense_20/Tanh:y:0*
T0*
_output_shapes
:	��
*batch_normalization_16/Cast/ReadVariableOpReadVariableOp3batch_normalization_16_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_16/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_16/batchnorm/addAddV24batch_normalization_16/Cast_1/ReadVariableOp:value:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_16/batchnorm/mulMuldense_20/SelectV2:output:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:	��
$batch_normalization_16/batchnorm/NegNeg2batch_normalization_16/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/mul_1Mul(batch_normalization_16/batchnorm/Neg:y:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/add_1AddV2(batch_normalization_16/batchnorm/mul:z:0*batch_normalization_16/batchnorm/mul_1:z:0*
T0*
_output_shapes
:	��
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMul*batch_normalization_16/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Z
dense_21/TanhTanhdense_21/BiasAdd:output:0*
T0*
_output_shapes
:	�T
dense_21/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    t
dense_21/LessLessdense_21/BiasAdd:output:0dense_21/Less/y:output:0*
T0*
_output_shapes
:	�S
dense_21/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>i
dense_21/MulMuldense_21/Tanh:y:0dense_21/Mul/y:output:0*
T0*
_output_shapes
:	�
dense_21/SelectV2SelectV2dense_21/Less:z:0dense_21/Mul:z:0dense_21/Tanh:y:0*
T0*
_output_shapes
:	��
*batch_normalization_17/Cast/ReadVariableOpReadVariableOp3batch_normalization_17_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_17/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_17_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_17/batchnorm/addAddV24batch_normalization_17/Cast_1/ReadVariableOp:value:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_17/batchnorm/mulMuldense_21/SelectV2:output:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:	��
$batch_normalization_17/batchnorm/NegNeg2batch_normalization_17/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/mul_1Mul(batch_normalization_17/batchnorm/Neg:y:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/add_1AddV2(batch_normalization_17/batchnorm/mul:z:0*batch_normalization_17/batchnorm/mul_1:z:0*
T0*
_output_shapes
:	��
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Z
dense_22/TanhTanhdense_22/BiasAdd:output:0*
T0*
_output_shapes
:	�T
dense_22/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    t
dense_22/LessLessdense_22/BiasAdd:output:0dense_22/Less/y:output:0*
T0*
_output_shapes
:	�S
dense_22/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>i
dense_22/MulMuldense_22/Tanh:y:0dense_22/Mul/y:output:0*
T0*
_output_shapes
:	�
dense_22/SelectV2SelectV2dense_22/Less:z:0dense_22/Mul:z:0dense_22/Tanh:y:0*
T0*
_output_shapes
:	��
*batch_normalization_18/Cast/ReadVariableOpReadVariableOp3batch_normalization_18_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_18/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_18_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_18/batchnorm/addAddV24batch_normalization_18/Cast_1/ReadVariableOp:value:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_18/batchnorm/mulMuldense_22/SelectV2:output:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:	��
$batch_normalization_18/batchnorm/NegNeg2batch_normalization_18/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/mul_1Mul(batch_normalization_18/batchnorm/Neg:y:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/add_1AddV2(batch_normalization_18/batchnorm/mul:z:0*batch_normalization_18/batchnorm/mul_1:z:0*
T0*
_output_shapes
:	��
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_23/MatMulMatMul*batch_normalization_18/batchnorm/add_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Z
dense_23/TanhTanhdense_23/BiasAdd:output:0*
T0*
_output_shapes
:	�T
dense_23/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    t
dense_23/LessLessdense_23/BiasAdd:output:0dense_23/Less/y:output:0*
T0*
_output_shapes
:	�S
dense_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>i
dense_23/MulMuldense_23/Tanh:y:0dense_23/Mul/y:output:0*
T0*
_output_shapes
:	�
dense_23/SelectV2SelectV2dense_23/Less:z:0dense_23/Mul:z:0dense_23/Tanh:y:0*
T0*
_output_shapes
:	��
*batch_normalization_19/Cast/ReadVariableOpReadVariableOp3batch_normalization_19_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_19/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_19_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_19/batchnorm/addAddV24batch_normalization_19/Cast_1/ReadVariableOp:value:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_19/batchnorm/mulMuldense_23/SelectV2:output:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:	��
$batch_normalization_19/batchnorm/NegNeg2batch_normalization_19/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/mul_1Mul(batch_normalization_19/batchnorm/Neg:y:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/add_1AddV2(batch_normalization_19/batchnorm/mul:z:0*batch_normalization_19/batchnorm/mul_1:z:0*
T0*
_output_shapes
:	��
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_24/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:<�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:<�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_25/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:<�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:<\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
clip_by_value/MinimumMinimumdense_25/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:<T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:<_
IdentityIdentitydense_24/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:<Y

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:<�
NoOpNoOp+^batch_normalization_16/Cast/ReadVariableOp-^batch_normalization_16/Cast_1/ReadVariableOp+^batch_normalization_17/Cast/ReadVariableOp-^batch_normalization_17/Cast_1/ReadVariableOp+^batch_normalization_18/Cast/ReadVariableOp-^batch_normalization_18/Cast_1/ReadVariableOp+^batch_normalization_19/Cast/ReadVariableOp-^batch_normalization_19/Cast_1/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:<: : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_16/Cast/ReadVariableOp*batch_normalization_16/Cast/ReadVariableOp2\
,batch_normalization_16/Cast_1/ReadVariableOp,batch_normalization_16/Cast_1/ReadVariableOp2X
*batch_normalization_17/Cast/ReadVariableOp*batch_normalization_17/Cast/ReadVariableOp2\
,batch_normalization_17/Cast_1/ReadVariableOp,batch_normalization_17/Cast_1/ReadVariableOp2X
*batch_normalization_18/Cast/ReadVariableOp*batch_normalization_18/Cast/ReadVariableOp2\
,batch_normalization_18/Cast_1/ReadVariableOp,batch_normalization_18/Cast_1/ReadVariableOp2X
*batch_normalization_19/Cast/ReadVariableOp*batch_normalization_19/Cast/ReadVariableOp2\
,batch_normalization_19/Cast_1/ReadVariableOp,batch_normalization_19/Cast_1/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:F B

_output_shapes

:<
 
_user_specified_nameinputs
݃
�
__inference_call_7397

inputs:
'dense_20_matmul_readvariableop_resource:	<�7
(dense_20_biasadd_readvariableop_resource:	�B
3batch_normalization_16_cast_readvariableop_resource:	�D
5batch_normalization_16_cast_1_readvariableop_resource:	�;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�B
3batch_normalization_17_cast_readvariableop_resource:	�D
5batch_normalization_17_cast_1_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�B
3batch_normalization_18_cast_readvariableop_resource:	�D
5batch_normalization_18_cast_1_readvariableop_resource:	�;
'dense_23_matmul_readvariableop_resource:
��7
(dense_23_biasadd_readvariableop_resource:	�B
3batch_normalization_19_cast_readvariableop_resource:	�D
5batch_normalization_19_cast_1_readvariableop_resource:	�:
'dense_24_matmul_readvariableop_resource:	�<6
(dense_24_biasadd_readvariableop_resource:<:
'dense_25_matmul_readvariableop_resource:	�<6
(dense_25_biasadd_readvariableop_resource:<
identity

identity_1��*batch_normalization_16/Cast/ReadVariableOp�,batch_normalization_16/Cast_1/ReadVariableOp�*batch_normalization_17/Cast/ReadVariableOp�,batch_normalization_17/Cast_1/ReadVariableOp�*batch_normalization_18/Cast/ReadVariableOp�,batch_normalization_18/Cast_1/ReadVariableOp�*batch_normalization_19/Cast/ReadVariableOp�,batch_normalization_19/Cast_1/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOpU
dense_20/CastCastinputs*

DstT0*

SrcT0*
_output_shapes

:<�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	<�*
dtype0~
dense_20/MatMulMatMuldense_20/Cast:y:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Z
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0*
_output_shapes
:	�T
dense_20/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    t
dense_20/LessLessdense_20/BiasAdd:output:0dense_20/Less/y:output:0*
T0*
_output_shapes
:	�S
dense_20/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>i
dense_20/MulMuldense_20/Tanh:y:0dense_20/Mul/y:output:0*
T0*
_output_shapes
:	�
dense_20/SelectV2SelectV2dense_20/Less:z:0dense_20/Mul:z:0dense_20/Tanh:y:0*
T0*
_output_shapes
:	��
*batch_normalization_16/Cast/ReadVariableOpReadVariableOp3batch_normalization_16_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_16/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_16/batchnorm/addAddV24batch_normalization_16/Cast_1/ReadVariableOp:value:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_16/batchnorm/mulMuldense_20/SelectV2:output:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:	��
$batch_normalization_16/batchnorm/NegNeg2batch_normalization_16/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/mul_1Mul(batch_normalization_16/batchnorm/Neg:y:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/add_1AddV2(batch_normalization_16/batchnorm/mul:z:0*batch_normalization_16/batchnorm/mul_1:z:0*
T0*
_output_shapes
:	��
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMul*batch_normalization_16/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Z
dense_21/TanhTanhdense_21/BiasAdd:output:0*
T0*
_output_shapes
:	�T
dense_21/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    t
dense_21/LessLessdense_21/BiasAdd:output:0dense_21/Less/y:output:0*
T0*
_output_shapes
:	�S
dense_21/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>i
dense_21/MulMuldense_21/Tanh:y:0dense_21/Mul/y:output:0*
T0*
_output_shapes
:	�
dense_21/SelectV2SelectV2dense_21/Less:z:0dense_21/Mul:z:0dense_21/Tanh:y:0*
T0*
_output_shapes
:	��
*batch_normalization_17/Cast/ReadVariableOpReadVariableOp3batch_normalization_17_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_17/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_17_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_17/batchnorm/addAddV24batch_normalization_17/Cast_1/ReadVariableOp:value:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_17/batchnorm/mulMuldense_21/SelectV2:output:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:	��
$batch_normalization_17/batchnorm/NegNeg2batch_normalization_17/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/mul_1Mul(batch_normalization_17/batchnorm/Neg:y:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/add_1AddV2(batch_normalization_17/batchnorm/mul:z:0*batch_normalization_17/batchnorm/mul_1:z:0*
T0*
_output_shapes
:	��
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Z
dense_22/TanhTanhdense_22/BiasAdd:output:0*
T0*
_output_shapes
:	�T
dense_22/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    t
dense_22/LessLessdense_22/BiasAdd:output:0dense_22/Less/y:output:0*
T0*
_output_shapes
:	�S
dense_22/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>i
dense_22/MulMuldense_22/Tanh:y:0dense_22/Mul/y:output:0*
T0*
_output_shapes
:	�
dense_22/SelectV2SelectV2dense_22/Less:z:0dense_22/Mul:z:0dense_22/Tanh:y:0*
T0*
_output_shapes
:	��
*batch_normalization_18/Cast/ReadVariableOpReadVariableOp3batch_normalization_18_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_18/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_18_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_18/batchnorm/addAddV24batch_normalization_18/Cast_1/ReadVariableOp:value:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_18/batchnorm/mulMuldense_22/SelectV2:output:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:	��
$batch_normalization_18/batchnorm/NegNeg2batch_normalization_18/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/mul_1Mul(batch_normalization_18/batchnorm/Neg:y:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/add_1AddV2(batch_normalization_18/batchnorm/mul:z:0*batch_normalization_18/batchnorm/mul_1:z:0*
T0*
_output_shapes
:	��
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_23/MatMulMatMul*batch_normalization_18/batchnorm/add_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Z
dense_23/TanhTanhdense_23/BiasAdd:output:0*
T0*
_output_shapes
:	�T
dense_23/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    t
dense_23/LessLessdense_23/BiasAdd:output:0dense_23/Less/y:output:0*
T0*
_output_shapes
:	�S
dense_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>i
dense_23/MulMuldense_23/Tanh:y:0dense_23/Mul/y:output:0*
T0*
_output_shapes
:	�
dense_23/SelectV2SelectV2dense_23/Less:z:0dense_23/Mul:z:0dense_23/Tanh:y:0*
T0*
_output_shapes
:	��
*batch_normalization_19/Cast/ReadVariableOpReadVariableOp3batch_normalization_19_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_19/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_19_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_19/batchnorm/addAddV24batch_normalization_19/Cast_1/ReadVariableOp:value:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_19/batchnorm/mulMuldense_23/SelectV2:output:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:	��
$batch_normalization_19/batchnorm/NegNeg2batch_normalization_19/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/mul_1Mul(batch_normalization_19/batchnorm/Neg:y:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/add_1AddV2(batch_normalization_19/batchnorm/mul:z:0*batch_normalization_19/batchnorm/mul_1:z:0*
T0*
_output_shapes
:	��
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_24/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:<�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:<�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_25/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:<�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:<\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
clip_by_value/MinimumMinimumdense_25/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:<T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:<_
IdentityIdentitydense_24/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:<Y

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:<�
NoOpNoOp+^batch_normalization_16/Cast/ReadVariableOp-^batch_normalization_16/Cast_1/ReadVariableOp+^batch_normalization_17/Cast/ReadVariableOp-^batch_normalization_17/Cast_1/ReadVariableOp+^batch_normalization_18/Cast/ReadVariableOp-^batch_normalization_18/Cast_1/ReadVariableOp+^batch_normalization_19/Cast/ReadVariableOp-^batch_normalization_19/Cast_1/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:<: : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_16/Cast/ReadVariableOp*batch_normalization_16/Cast/ReadVariableOp2\
,batch_normalization_16/Cast_1/ReadVariableOp,batch_normalization_16/Cast_1/ReadVariableOp2X
*batch_normalization_17/Cast/ReadVariableOp*batch_normalization_17/Cast/ReadVariableOp2\
,batch_normalization_17/Cast_1/ReadVariableOp,batch_normalization_17/Cast_1/ReadVariableOp2X
*batch_normalization_18/Cast/ReadVariableOp*batch_normalization_18/Cast/ReadVariableOp2\
,batch_normalization_18/Cast_1/ReadVariableOp,batch_normalization_18/Cast_1/ReadVariableOp2X
*batch_normalization_19/Cast/ReadVariableOp*batch_normalization_19/Cast/ReadVariableOp2\
,batch_normalization_19/Cast_1/ReadVariableOp,batch_normalization_19/Cast_1/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:F B

_output_shapes

:<
 
_user_specified_nameinputs
�5
�
1__inference_get_action_and_log_prob_density_27287	
state
unknown:	<�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�<

unknown_16:<

unknown_17:	�<

unknown_18:<
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	�<:	�<*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_call_27203V
ExpExp StatefulPartitionedCall:output:1*
T0*
_output_shapes
:	�<]
Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB n
Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   <   U
Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"   <   W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
:�
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
:g
Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*#
_output_shapes
:�<*
dtype0�
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:�<�
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:�<w
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Exp:y:0*
T0*#
_output_shapes
:�<�
Normal/sample/addAddV2Normal/sample/mul:z:0 StatefulPartitionedCall:output:0*
T0*#
_output_shapes
:�<l
Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   <   �
Normal/sample/ReshapeReshapeNormal/sample/add:z:0$Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	�<u
Normal/log_prob/truedivRealDivNormal/sample/Reshape:output:0Exp:y:0*
T0*
_output_shapes
:	�<y
Normal/log_prob/truediv_1RealDiv StatefulPartitionedCall:output:0Exp:y:0*
T0*
_output_shapes
:	�<�
!Normal/log_prob/SquaredDifferenceSquaredDifferenceNormal/log_prob/truediv:z:0Normal/log_prob/truediv_1:z:0*
T0*
_output_shapes
:	�<Z
Normal/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
Normal/log_prob/mulMulNormal/log_prob/mul/x:output:0%Normal/log_prob/SquaredDifference:z:0*
T0*
_output_shapes
:	�<Z
Normal/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�?k?M
Normal/log_prob/LogLogExp:y:0*
T0*
_output_shapes
:	�<
Normal/log_prob/addAddV2Normal/log_prob/Const:output:0Normal/log_prob/Log:y:0*
T0*
_output_shapes
:	�<v
Normal/log_prob/subSubNormal/log_prob/mul:z:0Normal/log_prob/add:z:0*
T0*
_output_shapes
:	�<e
IdentityIdentityNormal/sample/Reshape:output:0^NoOp*
T0*
_output_shapes
:	�<`

Identity_1IdentityNormal/log_prob/sub:z:0^NoOp*
T0*
_output_shapes
:	�<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:	�<: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes
:	�<

_user_specified_namestate
�	
�
F__inference_dense_24_layer_call_and_return_conditional_losses_15808301

inputs1
matmul_readvariableop_resource:	�<-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������<w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_dense_20_layer_call_and_return_conditional_losses_15808345

inputs1
matmul_readvariableop_resource:	<�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	<�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
LessLessBiasAdd:output:0Less/y:output:0*
T0*(
_output_shapes
:����������J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>W
MulMulTanh:y:0Mul/y:output:0*
T0*(
_output_shapes
:����������d
SelectV2SelectV2Less:z:0Mul:z:0Tanh:y:0*
T0*(
_output_shapes
:����������a
IdentityIdentitySelectV2:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�>
�

Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15807643

inputs$
dense_20_15807587:	<� 
dense_20_15807589:	�.
batch_normalization_16_15807592:	�.
batch_normalization_16_15807594:	�%
dense_21_15807597:
�� 
dense_21_15807599:	�.
batch_normalization_17_15807602:	�.
batch_normalization_17_15807604:	�%
dense_22_15807607:
�� 
dense_22_15807609:	�.
batch_normalization_18_15807612:	�.
batch_normalization_18_15807614:	�%
dense_23_15807617:
�� 
dense_23_15807619:	�.
batch_normalization_19_15807622:	�.
batch_normalization_19_15807624:	�$
dense_24_15807627:	�<
dense_24_15807629:<$
dense_25_15807632:	�<
dense_25_15807634:<
identity

identity_1��.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall�.batch_normalization_18/StatefulPartitionedCall�.batch_normalization_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall� dense_25/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinputsdense_20_15807587dense_20_15807589*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_20_layer_call_and_return_conditional_losses_15807300�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_16_15807592batch_normalization_16_15807594*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_15807078�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_21_15807597dense_21_15807599*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_21_layer_call_and_return_conditional_losses_15807327�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_17_15807602batch_normalization_17_15807604*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_15807142�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_22_15807607dense_22_15807609*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_22_layer_call_and_return_conditional_losses_15807354�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_18_15807612batch_normalization_18_15807614*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_18_layer_call_and_return_conditional_losses_15807206�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0dense_23_15807617dense_23_15807619*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_15807381�
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0batch_normalization_19_15807622batch_normalization_19_15807624*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_19_layer_call_and_return_conditional_losses_15807270�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0dense_24_15807627dense_24_15807629*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_15807402�
 dense_25/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0dense_25_15807632dense_25_15807634*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_15807418\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
clip_by_value/MinimumMinimum)dense_25/StatefulPartitionedCall:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������<T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������<x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<b

Identity_1Identityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:���������<�
NoOpNoOp/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������<: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
+__inference_dense_20_layer_call_fn_15808329

inputs
unknown:	<�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_20_layer_call_and_return_conditional_losses_15807300p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������<: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
+__inference_dense_25_layer_call_fn_15808310

inputs
unknown:	�<
	unknown_0:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_15807418o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�>
�

Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15807794
input_1$
dense_20_15807738:	<� 
dense_20_15807740:	�.
batch_normalization_16_15807743:	�.
batch_normalization_16_15807745:	�%
dense_21_15807748:
�� 
dense_21_15807750:	�.
batch_normalization_17_15807753:	�.
batch_normalization_17_15807755:	�%
dense_22_15807758:
�� 
dense_22_15807760:	�.
batch_normalization_18_15807763:	�.
batch_normalization_18_15807765:	�%
dense_23_15807768:
�� 
dense_23_15807770:	�.
batch_normalization_19_15807773:	�.
batch_normalization_19_15807775:	�$
dense_24_15807778:	�<
dense_24_15807780:<$
dense_25_15807783:	�<
dense_25_15807785:<
identity

identity_1��.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall�.batch_normalization_18/StatefulPartitionedCall�.batch_normalization_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall� dense_25/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_20_15807738dense_20_15807740*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_20_layer_call_and_return_conditional_losses_15807300�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_16_15807743batch_normalization_16_15807745*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_15807040�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_21_15807748dense_21_15807750*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_21_layer_call_and_return_conditional_losses_15807327�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_17_15807753batch_normalization_17_15807755*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_15807104�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_22_15807758dense_22_15807760*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_22_layer_call_and_return_conditional_losses_15807354�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_18_15807763batch_normalization_18_15807765*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_18_layer_call_and_return_conditional_losses_15807168�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0dense_23_15807768dense_23_15807770*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_15807381�
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0batch_normalization_19_15807773batch_normalization_19_15807775*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_19_layer_call_and_return_conditional_losses_15807232�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0dense_24_15807778dense_24_15807780*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_15807402�
 dense_25/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0dense_25_15807783dense_25_15807785*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_15807418\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
clip_by_value/MinimumMinimum)dense_25/StatefulPartitionedCall:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������<T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������<x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<b

Identity_1Identityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:���������<�
NoOpNoOp/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������<: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:P L
'
_output_shapes
:���������<
!
_user_specified_name	input_1
�
�
T__inference_batch_normalization_18_layer_call_and_return_conditional_losses_15807168

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�d
batchnorm/mulMulinputsbatchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:����������W
batchnorm/NegNegCast/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulbatchnorm/Neg:y:0batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul:z:0batchnorm/mul_1:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������t
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_18_layer_call_and_return_conditional_losses_15808606

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�d
batchnorm/mulMulinputsbatchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:����������T
batchnorm/NegNegmoments/Squeeze:output:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulbatchnorm/Neg:y:0batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul:z:0batchnorm/mul_1:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_15808544

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�d
batchnorm/mulMulinputsbatchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:����������T
batchnorm/NegNegmoments/Squeeze:output:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulbatchnorm/Neg:y:0batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul:z:0batchnorm/mul_1:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_18_layer_call_and_return_conditional_losses_15808577

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�d
batchnorm/mulMulinputsbatchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:����������W
batchnorm/NegNegCast/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulbatchnorm/Neg:y:0batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul:z:0batchnorm/mul_1:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������t
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_policy_network_soft_layer_call_fn_15807735
input_1
unknown:	<�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�<

unknown_16:<

unknown_17:	�<

unknown_18:<
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������<:���������<*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15807643o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������<: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������<
!
_user_specified_name	input_1
�7
�
0__inference_get_action_and_log_prob_density_7826	
state
unknown:	<�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�<

unknown_16:<

unknown_17:	�<

unknown_18:<
identity

identity_1��StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : a

ExpandDims
ExpandDimsstateExpandDims/dim:output:0*
T0*
_output_shapes

:<�
StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:<:<*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_call_7742U
ExpExp StatefulPartitionedCall:output:1*
T0*
_output_shapes

:<]
Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB n
Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   <   U
Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"   <   W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
:�
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
:g
Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*"
_output_shapes
:<*
dtype0�
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:<�
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:<v
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Exp:y:0*
T0*"
_output_shapes
:<�
Normal/sample/addAddV2Normal/sample/mul:z:0 StatefulPartitionedCall:output:0*
T0*"
_output_shapes
:<l
Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   <   �
Normal/sample/ReshapeReshapeNormal/sample/add:z:0$Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

:<t
Normal/log_prob/truedivRealDivNormal/sample/Reshape:output:0Exp:y:0*
T0*
_output_shapes

:<x
Normal/log_prob/truediv_1RealDiv StatefulPartitionedCall:output:0Exp:y:0*
T0*
_output_shapes

:<�
!Normal/log_prob/SquaredDifferenceSquaredDifferenceNormal/log_prob/truediv:z:0Normal/log_prob/truediv_1:z:0*
T0*
_output_shapes

:<Z
Normal/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
Normal/log_prob/mulMulNormal/log_prob/mul/x:output:0%Normal/log_prob/SquaredDifference:z:0*
T0*
_output_shapes

:<Z
Normal/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�?k?L
Normal/log_prob/LogLogExp:y:0*
T0*
_output_shapes

:<~
Normal/log_prob/addAddV2Normal/log_prob/Const:output:0Normal/log_prob/Log:y:0*
T0*
_output_shapes

:<u
Normal/log_prob/subSubNormal/log_prob/mul:z:0Normal/log_prob/add:z:0*
T0*
_output_shapes

:<d
IdentityIdentityNormal/sample/Reshape:output:0^NoOp*
T0*
_output_shapes

:<_

Identity_1IdentityNormal/log_prob/sub:z:0^NoOp*
T0*
_output_shapes

:<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:<: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:A =

_output_shapes
:<

_user_specified_namestate
�
�
9__inference_batch_normalization_18_layer_call_fn_15808562

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_18_layer_call_and_return_conditional_losses_15807206p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
݃
�
__inference_call_7511

inputs:
'dense_20_matmul_readvariableop_resource:	<�7
(dense_20_biasadd_readvariableop_resource:	�B
3batch_normalization_16_cast_readvariableop_resource:	�D
5batch_normalization_16_cast_1_readvariableop_resource:	�;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�B
3batch_normalization_17_cast_readvariableop_resource:	�D
5batch_normalization_17_cast_1_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�B
3batch_normalization_18_cast_readvariableop_resource:	�D
5batch_normalization_18_cast_1_readvariableop_resource:	�;
'dense_23_matmul_readvariableop_resource:
��7
(dense_23_biasadd_readvariableop_resource:	�B
3batch_normalization_19_cast_readvariableop_resource:	�D
5batch_normalization_19_cast_1_readvariableop_resource:	�:
'dense_24_matmul_readvariableop_resource:	�<6
(dense_24_biasadd_readvariableop_resource:<:
'dense_25_matmul_readvariableop_resource:	�<6
(dense_25_biasadd_readvariableop_resource:<
identity

identity_1��*batch_normalization_16/Cast/ReadVariableOp�,batch_normalization_16/Cast_1/ReadVariableOp�*batch_normalization_17/Cast/ReadVariableOp�,batch_normalization_17/Cast_1/ReadVariableOp�*batch_normalization_18/Cast/ReadVariableOp�,batch_normalization_18/Cast_1/ReadVariableOp�*batch_normalization_19/Cast/ReadVariableOp�,batch_normalization_19/Cast_1/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOpU
dense_20/CastCastinputs*

DstT0*

SrcT0*
_output_shapes

:<�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	<�*
dtype0~
dense_20/MatMulMatMuldense_20/Cast:y:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Z
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0*
_output_shapes
:	�T
dense_20/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    t
dense_20/LessLessdense_20/BiasAdd:output:0dense_20/Less/y:output:0*
T0*
_output_shapes
:	�S
dense_20/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>i
dense_20/MulMuldense_20/Tanh:y:0dense_20/Mul/y:output:0*
T0*
_output_shapes
:	�
dense_20/SelectV2SelectV2dense_20/Less:z:0dense_20/Mul:z:0dense_20/Tanh:y:0*
T0*
_output_shapes
:	��
*batch_normalization_16/Cast/ReadVariableOpReadVariableOp3batch_normalization_16_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_16/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_16/batchnorm/addAddV24batch_normalization_16/Cast_1/ReadVariableOp:value:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_16/batchnorm/mulMuldense_20/SelectV2:output:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:	��
$batch_normalization_16/batchnorm/NegNeg2batch_normalization_16/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/mul_1Mul(batch_normalization_16/batchnorm/Neg:y:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/add_1AddV2(batch_normalization_16/batchnorm/mul:z:0*batch_normalization_16/batchnorm/mul_1:z:0*
T0*
_output_shapes
:	��
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMul*batch_normalization_16/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Z
dense_21/TanhTanhdense_21/BiasAdd:output:0*
T0*
_output_shapes
:	�T
dense_21/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    t
dense_21/LessLessdense_21/BiasAdd:output:0dense_21/Less/y:output:0*
T0*
_output_shapes
:	�S
dense_21/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>i
dense_21/MulMuldense_21/Tanh:y:0dense_21/Mul/y:output:0*
T0*
_output_shapes
:	�
dense_21/SelectV2SelectV2dense_21/Less:z:0dense_21/Mul:z:0dense_21/Tanh:y:0*
T0*
_output_shapes
:	��
*batch_normalization_17/Cast/ReadVariableOpReadVariableOp3batch_normalization_17_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_17/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_17_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_17/batchnorm/addAddV24batch_normalization_17/Cast_1/ReadVariableOp:value:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_17/batchnorm/mulMuldense_21/SelectV2:output:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:	��
$batch_normalization_17/batchnorm/NegNeg2batch_normalization_17/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/mul_1Mul(batch_normalization_17/batchnorm/Neg:y:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/add_1AddV2(batch_normalization_17/batchnorm/mul:z:0*batch_normalization_17/batchnorm/mul_1:z:0*
T0*
_output_shapes
:	��
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Z
dense_22/TanhTanhdense_22/BiasAdd:output:0*
T0*
_output_shapes
:	�T
dense_22/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    t
dense_22/LessLessdense_22/BiasAdd:output:0dense_22/Less/y:output:0*
T0*
_output_shapes
:	�S
dense_22/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>i
dense_22/MulMuldense_22/Tanh:y:0dense_22/Mul/y:output:0*
T0*
_output_shapes
:	�
dense_22/SelectV2SelectV2dense_22/Less:z:0dense_22/Mul:z:0dense_22/Tanh:y:0*
T0*
_output_shapes
:	��
*batch_normalization_18/Cast/ReadVariableOpReadVariableOp3batch_normalization_18_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_18/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_18_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_18/batchnorm/addAddV24batch_normalization_18/Cast_1/ReadVariableOp:value:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_18/batchnorm/mulMuldense_22/SelectV2:output:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:	��
$batch_normalization_18/batchnorm/NegNeg2batch_normalization_18/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/mul_1Mul(batch_normalization_18/batchnorm/Neg:y:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/add_1AddV2(batch_normalization_18/batchnorm/mul:z:0*batch_normalization_18/batchnorm/mul_1:z:0*
T0*
_output_shapes
:	��
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_23/MatMulMatMul*batch_normalization_18/batchnorm/add_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Z
dense_23/TanhTanhdense_23/BiasAdd:output:0*
T0*
_output_shapes
:	�T
dense_23/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    t
dense_23/LessLessdense_23/BiasAdd:output:0dense_23/Less/y:output:0*
T0*
_output_shapes
:	�S
dense_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>i
dense_23/MulMuldense_23/Tanh:y:0dense_23/Mul/y:output:0*
T0*
_output_shapes
:	�
dense_23/SelectV2SelectV2dense_23/Less:z:0dense_23/Mul:z:0dense_23/Tanh:y:0*
T0*
_output_shapes
:	��
*batch_normalization_19/Cast/ReadVariableOpReadVariableOp3batch_normalization_19_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_19/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_19_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_19/batchnorm/addAddV24batch_normalization_19/Cast_1/ReadVariableOp:value:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_19/batchnorm/mulMuldense_23/SelectV2:output:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:	��
$batch_normalization_19/batchnorm/NegNeg2batch_normalization_19/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/mul_1Mul(batch_normalization_19/batchnorm/Neg:y:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/add_1AddV2(batch_normalization_19/batchnorm/mul:z:0*batch_normalization_19/batchnorm/mul_1:z:0*
T0*
_output_shapes
:	��
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_24/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:<�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:<�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_25/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:<�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:<\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
clip_by_value/MinimumMinimumdense_25/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:<T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:<_
IdentityIdentitydense_24/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:<Y

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:<�
NoOpNoOp+^batch_normalization_16/Cast/ReadVariableOp-^batch_normalization_16/Cast_1/ReadVariableOp+^batch_normalization_17/Cast/ReadVariableOp-^batch_normalization_17/Cast_1/ReadVariableOp+^batch_normalization_18/Cast/ReadVariableOp-^batch_normalization_18/Cast_1/ReadVariableOp+^batch_normalization_19/Cast/ReadVariableOp-^batch_normalization_19/Cast_1/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:<: : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_16/Cast/ReadVariableOp*batch_normalization_16/Cast/ReadVariableOp2\
,batch_normalization_16/Cast_1/ReadVariableOp,batch_normalization_16/Cast_1/ReadVariableOp2X
*batch_normalization_17/Cast/ReadVariableOp*batch_normalization_17/Cast/ReadVariableOp2\
,batch_normalization_17/Cast_1/ReadVariableOp,batch_normalization_17/Cast_1/ReadVariableOp2X
*batch_normalization_18/Cast/ReadVariableOp*batch_normalization_18/Cast/ReadVariableOp2\
,batch_normalization_18/Cast_1/ReadVariableOp,batch_normalization_18/Cast_1/ReadVariableOp2X
*batch_normalization_19/Cast/ReadVariableOp*batch_normalization_19/Cast/ReadVariableOp2\
,batch_normalization_19/Cast_1/ReadVariableOp,batch_normalization_19/Cast_1/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:F B

_output_shapes

:<
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_19_layer_call_and_return_conditional_losses_15807232

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�d
batchnorm/mulMulinputsbatchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:����������W
batchnorm/NegNegCast/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulbatchnorm/Neg:y:0batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul:z:0batchnorm/mul_1:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������t
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_18_layer_call_fn_15808553

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_18_layer_call_and_return_conditional_losses_15807168p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_dense_25_layer_call_and_return_conditional_losses_15808320

inputs1
matmul_readvariableop_resource:	�<-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������<w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_policy_network_soft_layer_call_fn_15807953

inputs
unknown:	<�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�<

unknown_16:<

unknown_17:	�<

unknown_18:<
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������<:���������<*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15807430o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������<: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
F__inference_dense_22_layer_call_and_return_conditional_losses_15807354

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
LessLessBiasAdd:output:0Less/y:output:0*
T0*(
_output_shapes
:����������J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>W
MulMulTanh:y:0Mul/y:output:0*
T0*(
_output_shapes
:����������d
SelectV2SelectV2Less:z:0Mul:z:0Tanh:y:0*
T0*(
_output_shapes
:����������a
IdentityIdentitySelectV2:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_16_layer_call_fn_15808438

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_15807078p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_dense_22_layer_call_and_return_conditional_losses_15808395

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
LessLessBiasAdd:output:0Less/y:output:0*
T0*(
_output_shapes
:����������J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>W
MulMulTanh:y:0Mul/y:output:0*
T0*(
_output_shapes
:����������d
SelectV2SelectV2Less:z:0Mul:z:0Tanh:y:0*
T0*(
_output_shapes
:����������a
IdentityIdentitySelectV2:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_16_layer_call_fn_15808429

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_15807040p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_15808482

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�d
batchnorm/mulMulinputsbatchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:����������T
batchnorm/NegNegmoments/Squeeze:output:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulbatchnorm/Neg:y:0batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul:z:0batchnorm/mul_1:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
__inference_call_7624

inputs:
'dense_20_matmul_readvariableop_resource:	<�7
(dense_20_biasadd_readvariableop_resource:	�B
3batch_normalization_16_cast_readvariableop_resource:	�D
5batch_normalization_16_cast_1_readvariableop_resource:	�;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�B
3batch_normalization_17_cast_readvariableop_resource:	�D
5batch_normalization_17_cast_1_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�B
3batch_normalization_18_cast_readvariableop_resource:	�D
5batch_normalization_18_cast_1_readvariableop_resource:	�;
'dense_23_matmul_readvariableop_resource:
��7
(dense_23_biasadd_readvariableop_resource:	�B
3batch_normalization_19_cast_readvariableop_resource:	�D
5batch_normalization_19_cast_1_readvariableop_resource:	�:
'dense_24_matmul_readvariableop_resource:	�<6
(dense_24_biasadd_readvariableop_resource:<:
'dense_25_matmul_readvariableop_resource:	�<6
(dense_25_biasadd_readvariableop_resource:<
identity

identity_1��*batch_normalization_16/Cast/ReadVariableOp�,batch_normalization_16/Cast_1/ReadVariableOp�*batch_normalization_17/Cast/ReadVariableOp�,batch_normalization_17/Cast_1/ReadVariableOp�*batch_normalization_18/Cast/ReadVariableOp�,batch_normalization_18/Cast_1/ReadVariableOp�*batch_normalization_19/Cast/ReadVariableOp�,batch_normalization_19/Cast_1/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	<�*
dtype0|
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0*(
_output_shapes
:����������T
dense_20/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
dense_20/LessLessdense_20/BiasAdd:output:0dense_20/Less/y:output:0*
T0*(
_output_shapes
:����������S
dense_20/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>r
dense_20/MulMuldense_20/Tanh:y:0dense_20/Mul/y:output:0*
T0*(
_output_shapes
:�����������
dense_20/SelectV2SelectV2dense_20/Less:z:0dense_20/Mul:z:0dense_20/Tanh:y:0*
T0*(
_output_shapes
:�����������
*batch_normalization_16/Cast/ReadVariableOpReadVariableOp3batch_normalization_16_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_16/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_16/batchnorm/addAddV24batch_normalization_16/Cast_1/ReadVariableOp:value:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_16/batchnorm/mulMuldense_20/SelectV2:output:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
$batch_normalization_16/batchnorm/NegNeg2batch_normalization_16/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/mul_1Mul(batch_normalization_16/batchnorm/Neg:y:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/add_1AddV2(batch_normalization_16/batchnorm/mul:z:0*batch_normalization_16/batchnorm/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMul*batch_normalization_16/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_21/TanhTanhdense_21/BiasAdd:output:0*
T0*(
_output_shapes
:����������T
dense_21/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
dense_21/LessLessdense_21/BiasAdd:output:0dense_21/Less/y:output:0*
T0*(
_output_shapes
:����������S
dense_21/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>r
dense_21/MulMuldense_21/Tanh:y:0dense_21/Mul/y:output:0*
T0*(
_output_shapes
:�����������
dense_21/SelectV2SelectV2dense_21/Less:z:0dense_21/Mul:z:0dense_21/Tanh:y:0*
T0*(
_output_shapes
:�����������
*batch_normalization_17/Cast/ReadVariableOpReadVariableOp3batch_normalization_17_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_17/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_17_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_17/batchnorm/addAddV24batch_normalization_17/Cast_1/ReadVariableOp:value:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_17/batchnorm/mulMuldense_21/SelectV2:output:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
$batch_normalization_17/batchnorm/NegNeg2batch_normalization_17/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/mul_1Mul(batch_normalization_17/batchnorm/Neg:y:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/add_1AddV2(batch_normalization_17/batchnorm/mul:z:0*batch_normalization_17/batchnorm/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_22/TanhTanhdense_22/BiasAdd:output:0*
T0*(
_output_shapes
:����������T
dense_22/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
dense_22/LessLessdense_22/BiasAdd:output:0dense_22/Less/y:output:0*
T0*(
_output_shapes
:����������S
dense_22/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>r
dense_22/MulMuldense_22/Tanh:y:0dense_22/Mul/y:output:0*
T0*(
_output_shapes
:�����������
dense_22/SelectV2SelectV2dense_22/Less:z:0dense_22/Mul:z:0dense_22/Tanh:y:0*
T0*(
_output_shapes
:�����������
*batch_normalization_18/Cast/ReadVariableOpReadVariableOp3batch_normalization_18_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_18/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_18_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_18/batchnorm/addAddV24batch_normalization_18/Cast_1/ReadVariableOp:value:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_18/batchnorm/mulMuldense_22/SelectV2:output:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
$batch_normalization_18/batchnorm/NegNeg2batch_normalization_18/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/mul_1Mul(batch_normalization_18/batchnorm/Neg:y:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/add_1AddV2(batch_normalization_18/batchnorm/mul:z:0*batch_normalization_18/batchnorm/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_23/MatMulMatMul*batch_normalization_18/batchnorm/add_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_23/TanhTanhdense_23/BiasAdd:output:0*
T0*(
_output_shapes
:����������T
dense_23/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
dense_23/LessLessdense_23/BiasAdd:output:0dense_23/Less/y:output:0*
T0*(
_output_shapes
:����������S
dense_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>r
dense_23/MulMuldense_23/Tanh:y:0dense_23/Mul/y:output:0*
T0*(
_output_shapes
:�����������
dense_23/SelectV2SelectV2dense_23/Less:z:0dense_23/Mul:z:0dense_23/Tanh:y:0*
T0*(
_output_shapes
:�����������
*batch_normalization_19/Cast/ReadVariableOpReadVariableOp3batch_normalization_19_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_19/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_19_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_19/batchnorm/addAddV24batch_normalization_19/Cast_1/ReadVariableOp:value:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_19/batchnorm/mulMuldense_23/SelectV2:output:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
$batch_normalization_19/batchnorm/NegNeg2batch_normalization_19/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/mul_1Mul(batch_normalization_19/batchnorm/Neg:y:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/add_1AddV2(batch_normalization_19/batchnorm/mul:z:0*batch_normalization_19/batchnorm/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_24/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_25/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
clip_by_value/MinimumMinimumdense_25/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������<T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������<h
IdentityIdentitydense_24/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������<b

Identity_1Identityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:���������<�
NoOpNoOp+^batch_normalization_16/Cast/ReadVariableOp-^batch_normalization_16/Cast_1/ReadVariableOp+^batch_normalization_17/Cast/ReadVariableOp-^batch_normalization_17/Cast_1/ReadVariableOp+^batch_normalization_18/Cast/ReadVariableOp-^batch_normalization_18/Cast_1/ReadVariableOp+^batch_normalization_19/Cast/ReadVariableOp-^batch_normalization_19/Cast_1/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������<: : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_16/Cast/ReadVariableOp*batch_normalization_16/Cast/ReadVariableOp2\
,batch_normalization_16/Cast_1/ReadVariableOp,batch_normalization_16/Cast_1/ReadVariableOp2X
*batch_normalization_17/Cast/ReadVariableOp*batch_normalization_17/Cast/ReadVariableOp2\
,batch_normalization_17/Cast_1/ReadVariableOp,batch_normalization_17/Cast_1/ReadVariableOp2X
*batch_normalization_18/Cast/ReadVariableOp*batch_normalization_18/Cast/ReadVariableOp2\
,batch_normalization_18/Cast_1/ReadVariableOp,batch_normalization_18/Cast_1/ReadVariableOp2X
*batch_normalization_19/Cast/ReadVariableOp*batch_normalization_19/Cast/ReadVariableOp2\
,batch_normalization_19/Cast_1/ReadVariableOp,batch_normalization_19/Cast_1/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_19_layer_call_and_return_conditional_losses_15807270

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�d
batchnorm/mulMulinputsbatchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:����������T
batchnorm/NegNegmoments/Squeeze:output:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulbatchnorm/Neg:y:0batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul:z:0batchnorm/mul_1:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
__inference_call_27203

inputs:
'dense_20_matmul_readvariableop_resource:	<�7
(dense_20_biasadd_readvariableop_resource:	�M
>batch_normalization_16_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_16_assignmovingavg_1_readvariableop_resource:	�;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�M
>batch_normalization_17_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_17_assignmovingavg_1_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�M
>batch_normalization_18_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_18_assignmovingavg_1_readvariableop_resource:	�;
'dense_23_matmul_readvariableop_resource:
��7
(dense_23_biasadd_readvariableop_resource:	�M
>batch_normalization_19_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_19_assignmovingavg_1_readvariableop_resource:	�:
'dense_24_matmul_readvariableop_resource:	�<6
(dense_24_biasadd_readvariableop_resource:<:
'dense_25_matmul_readvariableop_resource:	�<6
(dense_25_biasadd_readvariableop_resource:<
identity

identity_1��&batch_normalization_16/AssignMovingAvg�5batch_normalization_16/AssignMovingAvg/ReadVariableOp�(batch_normalization_16/AssignMovingAvg_1�7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp�&batch_normalization_17/AssignMovingAvg�5batch_normalization_17/AssignMovingAvg/ReadVariableOp�(batch_normalization_17/AssignMovingAvg_1�7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp�&batch_normalization_18/AssignMovingAvg�5batch_normalization_18/AssignMovingAvg/ReadVariableOp�(batch_normalization_18/AssignMovingAvg_1�7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp�&batch_normalization_19/AssignMovingAvg�5batch_normalization_19/AssignMovingAvg/ReadVariableOp�(batch_normalization_19/AssignMovingAvg_1�7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	<�*
dtype0t
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
���
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��[
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0* 
_output_shapes
:
��T
dense_20/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    u
dense_20/LessLessdense_20/BiasAdd:output:0dense_20/Less/y:output:0*
T0* 
_output_shapes
:
��S
dense_20/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>j
dense_20/MulMuldense_20/Tanh:y:0dense_20/Mul/y:output:0*
T0* 
_output_shapes
:
���
dense_20/SelectV2SelectV2dense_20/Less:z:0dense_20/Mul:z:0dense_20/Tanh:y:0*
T0* 
_output_shapes
:
��
5batch_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_16/moments/meanMeandense_20/SelectV2:output:0>batch_normalization_16/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_16/moments/StopGradientStopGradient,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_16/moments/SquaredDifferenceSquaredDifferencedense_20/SelectV2:output:04batch_normalization_16/moments/StopGradient:output:0*
T0* 
_output_shapes
:
���
9batch_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_16/moments/varianceMean4batch_normalization_16/moments/SquaredDifference:z:0Bbatch_normalization_16/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_16/moments/SqueezeSqueeze,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_16/moments/Squeeze_1Squeeze0batch_normalization_16/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_16/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_16_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_16/AssignMovingAvg/subSub=batch_normalization_16/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_16/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_16/AssignMovingAvg/mulMul.batch_normalization_16/AssignMovingAvg/sub:z:05batch_normalization_16/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_16/AssignMovingAvgAssignSubVariableOp>batch_normalization_16_assignmovingavg_readvariableop_resource.batch_normalization_16/AssignMovingAvg/mul:z:06^batch_normalization_16/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_16/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_16_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_16/AssignMovingAvg_1/subSub?batch_normalization_16/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_16/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_16/AssignMovingAvg_1/mulMul0batch_normalization_16/AssignMovingAvg_1/sub:z:07batch_normalization_16/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_16/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_16_assignmovingavg_1_readvariableop_resource0batch_normalization_16/AssignMovingAvg_1/mul:z:08^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_16/batchnorm/addAddV21batch_normalization_16/moments/Squeeze_1:output:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_16/batchnorm/mulMuldense_20/SelectV2:output:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0* 
_output_shapes
:
���
$batch_normalization_16/batchnorm/NegNeg/batch_normalization_16/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/mul_1Mul(batch_normalization_16/batchnorm/Neg:y:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/add_1AddV2(batch_normalization_16/batchnorm/mul:z:0*batch_normalization_16/batchnorm/mul_1:z:0*
T0* 
_output_shapes
:
���
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMul*batch_normalization_16/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
���
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��[
dense_21/TanhTanhdense_21/BiasAdd:output:0*
T0* 
_output_shapes
:
��T
dense_21/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    u
dense_21/LessLessdense_21/BiasAdd:output:0dense_21/Less/y:output:0*
T0* 
_output_shapes
:
��S
dense_21/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>j
dense_21/MulMuldense_21/Tanh:y:0dense_21/Mul/y:output:0*
T0* 
_output_shapes
:
���
dense_21/SelectV2SelectV2dense_21/Less:z:0dense_21/Mul:z:0dense_21/Tanh:y:0*
T0* 
_output_shapes
:
��
5batch_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_17/moments/meanMeandense_21/SelectV2:output:0>batch_normalization_17/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_17/moments/StopGradientStopGradient,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_17/moments/SquaredDifferenceSquaredDifferencedense_21/SelectV2:output:04batch_normalization_17/moments/StopGradient:output:0*
T0* 
_output_shapes
:
���
9batch_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_17/moments/varianceMean4batch_normalization_17/moments/SquaredDifference:z:0Bbatch_normalization_17/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_17/moments/SqueezeSqueeze,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_17/moments/Squeeze_1Squeeze0batch_normalization_17/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_17/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_17/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_17_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_17/AssignMovingAvg/subSub=batch_normalization_17/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_17/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_17/AssignMovingAvg/mulMul.batch_normalization_17/AssignMovingAvg/sub:z:05batch_normalization_17/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_17/AssignMovingAvgAssignSubVariableOp>batch_normalization_17_assignmovingavg_readvariableop_resource.batch_normalization_17/AssignMovingAvg/mul:z:06^batch_normalization_17/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_17/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_17_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_17/AssignMovingAvg_1/subSub?batch_normalization_17/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_17/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_17/AssignMovingAvg_1/mulMul0batch_normalization_17/AssignMovingAvg_1/sub:z:07batch_normalization_17/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_17/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_17_assignmovingavg_1_readvariableop_resource0batch_normalization_17/AssignMovingAvg_1/mul:z:08^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_17/batchnorm/addAddV21batch_normalization_17/moments/Squeeze_1:output:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_17/batchnorm/mulMuldense_21/SelectV2:output:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0* 
_output_shapes
:
���
$batch_normalization_17/batchnorm/NegNeg/batch_normalization_17/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/mul_1Mul(batch_normalization_17/batchnorm/Neg:y:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/add_1AddV2(batch_normalization_17/batchnorm/mul:z:0*batch_normalization_17/batchnorm/mul_1:z:0*
T0* 
_output_shapes
:
���
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
���
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��[
dense_22/TanhTanhdense_22/BiasAdd:output:0*
T0* 
_output_shapes
:
��T
dense_22/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    u
dense_22/LessLessdense_22/BiasAdd:output:0dense_22/Less/y:output:0*
T0* 
_output_shapes
:
��S
dense_22/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>j
dense_22/MulMuldense_22/Tanh:y:0dense_22/Mul/y:output:0*
T0* 
_output_shapes
:
���
dense_22/SelectV2SelectV2dense_22/Less:z:0dense_22/Mul:z:0dense_22/Tanh:y:0*
T0* 
_output_shapes
:
��
5batch_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_18/moments/meanMeandense_22/SelectV2:output:0>batch_normalization_18/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_18/moments/StopGradientStopGradient,batch_normalization_18/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_18/moments/SquaredDifferenceSquaredDifferencedense_22/SelectV2:output:04batch_normalization_18/moments/StopGradient:output:0*
T0* 
_output_shapes
:
���
9batch_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_18/moments/varianceMean4batch_normalization_18/moments/SquaredDifference:z:0Bbatch_normalization_18/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_18/moments/SqueezeSqueeze,batch_normalization_18/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_18/moments/Squeeze_1Squeeze0batch_normalization_18/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_18/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_18/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_18_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_18/AssignMovingAvg/subSub=batch_normalization_18/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_18/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_18/AssignMovingAvg/mulMul.batch_normalization_18/AssignMovingAvg/sub:z:05batch_normalization_18/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_18/AssignMovingAvgAssignSubVariableOp>batch_normalization_18_assignmovingavg_readvariableop_resource.batch_normalization_18/AssignMovingAvg/mul:z:06^batch_normalization_18/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_18/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_18_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_18/AssignMovingAvg_1/subSub?batch_normalization_18/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_18/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_18/AssignMovingAvg_1/mulMul0batch_normalization_18/AssignMovingAvg_1/sub:z:07batch_normalization_18/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_18/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_18_assignmovingavg_1_readvariableop_resource0batch_normalization_18/AssignMovingAvg_1/mul:z:08^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_18/batchnorm/addAddV21batch_normalization_18/moments/Squeeze_1:output:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_18/batchnorm/mulMuldense_22/SelectV2:output:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0* 
_output_shapes
:
���
$batch_normalization_18/batchnorm/NegNeg/batch_normalization_18/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/mul_1Mul(batch_normalization_18/batchnorm/Neg:y:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/add_1AddV2(batch_normalization_18/batchnorm/mul:z:0*batch_normalization_18/batchnorm/mul_1:z:0*
T0* 
_output_shapes
:
���
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_23/MatMulMatMul*batch_normalization_18/batchnorm/add_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
���
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��[
dense_23/TanhTanhdense_23/BiasAdd:output:0*
T0* 
_output_shapes
:
��T
dense_23/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    u
dense_23/LessLessdense_23/BiasAdd:output:0dense_23/Less/y:output:0*
T0* 
_output_shapes
:
��S
dense_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>j
dense_23/MulMuldense_23/Tanh:y:0dense_23/Mul/y:output:0*
T0* 
_output_shapes
:
���
dense_23/SelectV2SelectV2dense_23/Less:z:0dense_23/Mul:z:0dense_23/Tanh:y:0*
T0* 
_output_shapes
:
��
5batch_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_19/moments/meanMeandense_23/SelectV2:output:0>batch_normalization_19/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_19/moments/StopGradientStopGradient,batch_normalization_19/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_19/moments/SquaredDifferenceSquaredDifferencedense_23/SelectV2:output:04batch_normalization_19/moments/StopGradient:output:0*
T0* 
_output_shapes
:
���
9batch_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_19/moments/varianceMean4batch_normalization_19/moments/SquaredDifference:z:0Bbatch_normalization_19/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_19/moments/SqueezeSqueeze,batch_normalization_19/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_19/moments/Squeeze_1Squeeze0batch_normalization_19/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_19/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_19/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_19_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_19/AssignMovingAvg/subSub=batch_normalization_19/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_19/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_19/AssignMovingAvg/mulMul.batch_normalization_19/AssignMovingAvg/sub:z:05batch_normalization_19/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_19/AssignMovingAvgAssignSubVariableOp>batch_normalization_19_assignmovingavg_readvariableop_resource.batch_normalization_19/AssignMovingAvg/mul:z:06^batch_normalization_19/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_19/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_19_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_19/AssignMovingAvg_1/subSub?batch_normalization_19/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_19/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_19/AssignMovingAvg_1/mulMul0batch_normalization_19/AssignMovingAvg_1/sub:z:07batch_normalization_19/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_19/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_19_assignmovingavg_1_readvariableop_resource0batch_normalization_19/AssignMovingAvg_1/mul:z:08^batch_normalization_19/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_19/batchnorm/addAddV21batch_normalization_19/moments/Squeeze_1:output:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_19/batchnorm/mulMuldense_23/SelectV2:output:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0* 
_output_shapes
:
���
$batch_normalization_19/batchnorm/NegNeg/batch_normalization_19/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/mul_1Mul(batch_normalization_19/batchnorm/Neg:y:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/add_1AddV2(batch_normalization_19/batchnorm/mul:z:0*batch_normalization_19/batchnorm/mul_1:z:0*
T0* 
_output_shapes
:
���
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_24/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�<�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�<�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_25/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�<�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�<\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
clip_by_value/MinimumMinimumdense_25/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:	�<T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��w
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:	�<`
IdentityIdentitydense_24/BiasAdd:output:0^NoOp*
T0*
_output_shapes
:	�<Z

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes
:	�<�	
NoOpNoOp'^batch_normalization_16/AssignMovingAvg6^batch_normalization_16/AssignMovingAvg/ReadVariableOp)^batch_normalization_16/AssignMovingAvg_18^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp'^batch_normalization_17/AssignMovingAvg6^batch_normalization_17/AssignMovingAvg/ReadVariableOp)^batch_normalization_17/AssignMovingAvg_18^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp'^batch_normalization_18/AssignMovingAvg6^batch_normalization_18/AssignMovingAvg/ReadVariableOp)^batch_normalization_18/AssignMovingAvg_18^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp'^batch_normalization_19/AssignMovingAvg6^batch_normalization_19/AssignMovingAvg/ReadVariableOp)^batch_normalization_19/AssignMovingAvg_18^batch_normalization_19/AssignMovingAvg_1/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:	�<: : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_16/AssignMovingAvg&batch_normalization_16/AssignMovingAvg2n
5batch_normalization_16/AssignMovingAvg/ReadVariableOp5batch_normalization_16/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_16/AssignMovingAvg_1(batch_normalization_16/AssignMovingAvg_12r
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp2P
&batch_normalization_17/AssignMovingAvg&batch_normalization_17/AssignMovingAvg2n
5batch_normalization_17/AssignMovingAvg/ReadVariableOp5batch_normalization_17/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_17/AssignMovingAvg_1(batch_normalization_17/AssignMovingAvg_12r
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp2P
&batch_normalization_18/AssignMovingAvg&batch_normalization_18/AssignMovingAvg2n
5batch_normalization_18/AssignMovingAvg/ReadVariableOp5batch_normalization_18/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_18/AssignMovingAvg_1(batch_normalization_18/AssignMovingAvg_12r
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp2P
&batch_normalization_19/AssignMovingAvg&batch_normalization_19/AssignMovingAvg2n
5batch_normalization_19/AssignMovingAvg/ReadVariableOp5batch_normalization_19/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_19/AssignMovingAvg_1(batch_normalization_19/AssignMovingAvg_12r
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:G C

_output_shapes
:	�<
 
_user_specified_nameinputs
�
�
F__inference_dense_20_layer_call_and_return_conditional_losses_15807300

inputs1
matmul_readvariableop_resource:	<�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	<�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
LessLessBiasAdd:output:0Less/y:output:0*
T0*(
_output_shapes
:����������J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>W
MulMulTanh:y:0Mul/y:output:0*
T0*(
_output_shapes
:����������d
SelectV2SelectV2Less:z:0Mul:z:0Tanh:y:0*
T0*(
_output_shapes
:����������a
IdentityIdentitySelectV2:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
��
�
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15808113

inputs:
'dense_20_matmul_readvariableop_resource:	<�7
(dense_20_biasadd_readvariableop_resource:	�B
3batch_normalization_16_cast_readvariableop_resource:	�D
5batch_normalization_16_cast_1_readvariableop_resource:	�;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�B
3batch_normalization_17_cast_readvariableop_resource:	�D
5batch_normalization_17_cast_1_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�B
3batch_normalization_18_cast_readvariableop_resource:	�D
5batch_normalization_18_cast_1_readvariableop_resource:	�;
'dense_23_matmul_readvariableop_resource:
��7
(dense_23_biasadd_readvariableop_resource:	�B
3batch_normalization_19_cast_readvariableop_resource:	�D
5batch_normalization_19_cast_1_readvariableop_resource:	�:
'dense_24_matmul_readvariableop_resource:	�<6
(dense_24_biasadd_readvariableop_resource:<:
'dense_25_matmul_readvariableop_resource:	�<6
(dense_25_biasadd_readvariableop_resource:<
identity

identity_1��*batch_normalization_16/Cast/ReadVariableOp�,batch_normalization_16/Cast_1/ReadVariableOp�*batch_normalization_17/Cast/ReadVariableOp�,batch_normalization_17/Cast_1/ReadVariableOp�*batch_normalization_18/Cast/ReadVariableOp�,batch_normalization_18/Cast_1/ReadVariableOp�*batch_normalization_19/Cast/ReadVariableOp�,batch_normalization_19/Cast_1/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	<�*
dtype0|
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0*(
_output_shapes
:����������T
dense_20/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
dense_20/LessLessdense_20/BiasAdd:output:0dense_20/Less/y:output:0*
T0*(
_output_shapes
:����������S
dense_20/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>r
dense_20/MulMuldense_20/Tanh:y:0dense_20/Mul/y:output:0*
T0*(
_output_shapes
:�����������
dense_20/SelectV2SelectV2dense_20/Less:z:0dense_20/Mul:z:0dense_20/Tanh:y:0*
T0*(
_output_shapes
:�����������
*batch_normalization_16/Cast/ReadVariableOpReadVariableOp3batch_normalization_16_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_16/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_16/batchnorm/addAddV24batch_normalization_16/Cast_1/ReadVariableOp:value:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_16/batchnorm/mulMuldense_20/SelectV2:output:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
$batch_normalization_16/batchnorm/NegNeg2batch_normalization_16/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/mul_1Mul(batch_normalization_16/batchnorm/Neg:y:0*batch_normalization_16/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/add_1AddV2(batch_normalization_16/batchnorm/mul:z:0*batch_normalization_16/batchnorm/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMul*batch_normalization_16/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_21/TanhTanhdense_21/BiasAdd:output:0*
T0*(
_output_shapes
:����������T
dense_21/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
dense_21/LessLessdense_21/BiasAdd:output:0dense_21/Less/y:output:0*
T0*(
_output_shapes
:����������S
dense_21/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>r
dense_21/MulMuldense_21/Tanh:y:0dense_21/Mul/y:output:0*
T0*(
_output_shapes
:�����������
dense_21/SelectV2SelectV2dense_21/Less:z:0dense_21/Mul:z:0dense_21/Tanh:y:0*
T0*(
_output_shapes
:�����������
*batch_normalization_17/Cast/ReadVariableOpReadVariableOp3batch_normalization_17_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_17/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_17_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_17/batchnorm/addAddV24batch_normalization_17/Cast_1/ReadVariableOp:value:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_17/batchnorm/mulMuldense_21/SelectV2:output:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
$batch_normalization_17/batchnorm/NegNeg2batch_normalization_17/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/mul_1Mul(batch_normalization_17/batchnorm/Neg:y:0*batch_normalization_17/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/add_1AddV2(batch_normalization_17/batchnorm/mul:z:0*batch_normalization_17/batchnorm/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_22/TanhTanhdense_22/BiasAdd:output:0*
T0*(
_output_shapes
:����������T
dense_22/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
dense_22/LessLessdense_22/BiasAdd:output:0dense_22/Less/y:output:0*
T0*(
_output_shapes
:����������S
dense_22/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>r
dense_22/MulMuldense_22/Tanh:y:0dense_22/Mul/y:output:0*
T0*(
_output_shapes
:�����������
dense_22/SelectV2SelectV2dense_22/Less:z:0dense_22/Mul:z:0dense_22/Tanh:y:0*
T0*(
_output_shapes
:�����������
*batch_normalization_18/Cast/ReadVariableOpReadVariableOp3batch_normalization_18_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_18/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_18_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_18/batchnorm/addAddV24batch_normalization_18/Cast_1/ReadVariableOp:value:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_18/batchnorm/mulMuldense_22/SelectV2:output:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
$batch_normalization_18/batchnorm/NegNeg2batch_normalization_18/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/mul_1Mul(batch_normalization_18/batchnorm/Neg:y:0*batch_normalization_18/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/add_1AddV2(batch_normalization_18/batchnorm/mul:z:0*batch_normalization_18/batchnorm/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_23/MatMulMatMul*batch_normalization_18/batchnorm/add_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_23/TanhTanhdense_23/BiasAdd:output:0*
T0*(
_output_shapes
:����������T
dense_23/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
dense_23/LessLessdense_23/BiasAdd:output:0dense_23/Less/y:output:0*
T0*(
_output_shapes
:����������S
dense_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>r
dense_23/MulMuldense_23/Tanh:y:0dense_23/Mul/y:output:0*
T0*(
_output_shapes
:�����������
dense_23/SelectV2SelectV2dense_23/Less:z:0dense_23/Mul:z:0dense_23/Tanh:y:0*
T0*(
_output_shapes
:�����������
*batch_normalization_19/Cast/ReadVariableOpReadVariableOp3batch_normalization_19_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_19/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_19_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_19/batchnorm/addAddV24batch_normalization_19/Cast_1/ReadVariableOp:value:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_19/batchnorm/mulMuldense_23/SelectV2:output:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
$batch_normalization_19/batchnorm/NegNeg2batch_normalization_19/Cast/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/mul_1Mul(batch_normalization_19/batchnorm/Neg:y:0*batch_normalization_19/batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/add_1AddV2(batch_normalization_19/batchnorm/mul:z:0*batch_normalization_19/batchnorm/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_24/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	�<*
dtype0�
dense_25/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
clip_by_value/MinimumMinimumdense_25/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������<T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������<h
IdentityIdentitydense_24/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������<b

Identity_1Identityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:���������<�
NoOpNoOp+^batch_normalization_16/Cast/ReadVariableOp-^batch_normalization_16/Cast_1/ReadVariableOp+^batch_normalization_17/Cast/ReadVariableOp-^batch_normalization_17/Cast_1/ReadVariableOp+^batch_normalization_18/Cast/ReadVariableOp-^batch_normalization_18/Cast_1/ReadVariableOp+^batch_normalization_19/Cast/ReadVariableOp-^batch_normalization_19/Cast_1/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������<: : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_16/Cast/ReadVariableOp*batch_normalization_16/Cast/ReadVariableOp2\
,batch_normalization_16/Cast_1/ReadVariableOp,batch_normalization_16/Cast_1/ReadVariableOp2X
*batch_normalization_17/Cast/ReadVariableOp*batch_normalization_17/Cast/ReadVariableOp2\
,batch_normalization_17/Cast_1/ReadVariableOp,batch_normalization_17/Cast_1/ReadVariableOp2X
*batch_normalization_18/Cast/ReadVariableOp*batch_normalization_18/Cast/ReadVariableOp2\
,batch_normalization_18/Cast_1/ReadVariableOp,batch_normalization_18/Cast_1/ReadVariableOp2X
*batch_normalization_19/Cast/ReadVariableOp*batch_normalization_19/Cast/ReadVariableOp2\
,batch_normalization_19/Cast_1/ReadVariableOp,batch_normalization_19/Cast_1/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_18_layer_call_and_return_conditional_losses_15807206

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�d
batchnorm/mulMulinputsbatchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:����������T
batchnorm/NegNegmoments/Squeeze:output:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulbatchnorm/Neg:y:0batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul:z:0batchnorm/mul_1:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_19_layer_call_fn_15808615

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_19_layer_call_and_return_conditional_losses_15807232p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_15808453

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�d
batchnorm/mulMulinputsbatchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:����������W
batchnorm/NegNegCast/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulbatchnorm/Neg:y:0batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul:z:0batchnorm/mul_1:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������t
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_22_layer_call_fn_15808379

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_22_layer_call_and_return_conditional_losses_15807354p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_15807104

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�d
batchnorm/mulMulinputsbatchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:����������W
batchnorm/NegNegCast/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulbatchnorm/Neg:y:0batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul:z:0batchnorm/mul_1:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������t
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_15807040

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�d
batchnorm/mulMulinputsbatchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:����������W
batchnorm/NegNegCast/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulbatchnorm/Neg:y:0batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul:z:0batchnorm/mul_1:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������t
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�'
$__inference__traced_restore_15809062
file_prefix3
 assignvariableop_dense_20_kernel:	<�/
 assignvariableop_1_dense_20_bias:	�6
"assignvariableop_2_dense_21_kernel:
��/
 assignvariableop_3_dense_21_bias:	�6
"assignvariableop_4_dense_22_kernel:
��/
 assignvariableop_5_dense_22_bias:	�6
"assignvariableop_6_dense_23_kernel:
��/
 assignvariableop_7_dense_23_bias:	�D
5assignvariableop_8_batch_normalization_16_moving_mean:	�H
9assignvariableop_9_batch_normalization_16_moving_variance:	�E
6assignvariableop_10_batch_normalization_17_moving_mean:	�I
:assignvariableop_11_batch_normalization_17_moving_variance:	�E
6assignvariableop_12_batch_normalization_18_moving_mean:	�I
:assignvariableop_13_batch_normalization_18_moving_variance:	�E
6assignvariableop_14_batch_normalization_19_moving_mean:	�I
:assignvariableop_15_batch_normalization_19_moving_variance:	�6
#assignvariableop_16_dense_24_kernel:	�</
!assignvariableop_17_dense_24_bias:<6
#assignvariableop_18_dense_25_kernel:	�</
!assignvariableop_19_dense_25_bias:<$
assignvariableop_20_beta_1: $
assignvariableop_21_beta_2: #
assignvariableop_22_decay: "
assignvariableop_23_iter:	 ?
,assignvariableop_24_adam_2_dense_20_kernel_m:	<�9
*assignvariableop_25_adam_2_dense_20_bias_m:	�@
,assignvariableop_26_adam_2_dense_21_kernel_m:
��9
*assignvariableop_27_adam_2_dense_21_bias_m:	�@
,assignvariableop_28_adam_2_dense_22_kernel_m:
��9
*assignvariableop_29_adam_2_dense_22_bias_m:	�@
,assignvariableop_30_adam_2_dense_23_kernel_m:
��9
*assignvariableop_31_adam_2_dense_23_bias_m:	�?
,assignvariableop_32_adam_2_dense_24_kernel_m:	�<8
*assignvariableop_33_adam_2_dense_24_bias_m:<?
,assignvariableop_34_adam_2_dense_25_kernel_m:	�<8
*assignvariableop_35_adam_2_dense_25_bias_m:<?
,assignvariableop_36_adam_2_dense_20_kernel_v:	<�9
*assignvariableop_37_adam_2_dense_20_bias_v:	�@
,assignvariableop_38_adam_2_dense_21_kernel_v:
��9
*assignvariableop_39_adam_2_dense_21_bias_v:	�@
,assignvariableop_40_adam_2_dense_22_kernel_v:
��9
*assignvariableop_41_adam_2_dense_22_bias_v:	�@
,assignvariableop_42_adam_2_dense_23_kernel_v:
��9
*assignvariableop_43_adam_2_dense_23_bias_v:	�?
,assignvariableop_44_adam_2_dense_24_kernel_v:	�<8
*assignvariableop_45_adam_2_dense_24_bias_v:<?
,assignvariableop_46_adam_2_dense_25_kernel_v:	�<8
*assignvariableop_47_adam_2_dense_25_bias_v:<B
/assignvariableop_48_adam_2_dense_20_kernel_vhat:	<�<
-assignvariableop_49_adam_2_dense_20_bias_vhat:	�C
/assignvariableop_50_adam_2_dense_21_kernel_vhat:
��<
-assignvariableop_51_adam_2_dense_21_bias_vhat:	�C
/assignvariableop_52_adam_2_dense_22_kernel_vhat:
��<
-assignvariableop_53_adam_2_dense_22_bias_vhat:	�C
/assignvariableop_54_adam_2_dense_23_kernel_vhat:
��<
-assignvariableop_55_adam_2_dense_23_bias_vhat:	�B
/assignvariableop_56_adam_2_dense_24_kernel_vhat:	�<;
-assignvariableop_57_adam_2_dense_24_bias_vhat:<B
/assignvariableop_58_adam_2_dense_25_kernel_vhat:	�<;
-assignvariableop_59_adam_2_dense_25_bias_vhat:<
identity_61��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/6/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/7/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBFvariables/16/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBFvariables/17/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBFvariables/18/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBFvariables/19/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*K
dtypesA
?2=	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_20_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_20_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_21_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_21_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_22_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_22_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_23_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_23_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_16_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_16_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_17_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_17_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp6assignvariableop_12_batch_normalization_18_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp:assignvariableop_13_batch_normalization_18_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp6assignvariableop_14_batch_normalization_19_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp:assignvariableop_15_batch_normalization_19_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_24_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_24_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_25_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_25_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_beta_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_beta_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_decayIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_iterIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp,assignvariableop_24_adam_2_dense_20_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_2_dense_20_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_2_dense_21_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_2_dense_21_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp,assignvariableop_28_adam_2_dense_22_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_2_dense_22_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp,assignvariableop_30_adam_2_dense_23_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_2_dense_23_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_2_dense_24_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_2_dense_24_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp,assignvariableop_34_adam_2_dense_25_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_2_dense_25_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_2_dense_20_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_2_dense_20_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp,assignvariableop_38_adam_2_dense_21_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_2_dense_21_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp,assignvariableop_40_adam_2_dense_22_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_2_dense_22_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp,assignvariableop_42_adam_2_dense_23_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_2_dense_23_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp,assignvariableop_44_adam_2_dense_24_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_2_dense_24_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp,assignvariableop_46_adam_2_dense_25_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_2_dense_25_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp/assignvariableop_48_adam_2_dense_20_kernel_vhatIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp-assignvariableop_49_adam_2_dense_20_bias_vhatIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp/assignvariableop_50_adam_2_dense_21_kernel_vhatIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp-assignvariableop_51_adam_2_dense_21_bias_vhatIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp/assignvariableop_52_adam_2_dense_22_kernel_vhatIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp-assignvariableop_53_adam_2_dense_22_bias_vhatIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp/assignvariableop_54_adam_2_dense_23_kernel_vhatIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp-assignvariableop_55_adam_2_dense_23_bias_vhatIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp/assignvariableop_56_adam_2_dense_24_kernel_vhatIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp-assignvariableop_57_adam_2_dense_24_bias_vhatIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp/assignvariableop_58_adam_2_dense_25_kernel_vhatIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp-assignvariableop_59_adam_2_dense_25_bias_vhatIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_60Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_61IdentityIdentity_60:output:0^NoOp_1*
T0*
_output_shapes
: �

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_61Identity_61:output:0*�
_input_shapes|
z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
+__inference_dense_23_layer_call_fn_15808404

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_15807381p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_15807078

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�d
batchnorm/mulMulinputsbatchnorm/Rsqrt:y:0*
T0*(
_output_shapes
:����������T
batchnorm/NegNegmoments/Squeeze:output:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulbatchnorm/Neg:y:0batchnorm/Rsqrt:y:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul:z:0batchnorm/mul_1:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�>
�

Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15807430

inputs$
dense_20_15807301:	<� 
dense_20_15807303:	�.
batch_normalization_16_15807306:	�.
batch_normalization_16_15807308:	�%
dense_21_15807328:
�� 
dense_21_15807330:	�.
batch_normalization_17_15807333:	�.
batch_normalization_17_15807335:	�%
dense_22_15807355:
�� 
dense_22_15807357:	�.
batch_normalization_18_15807360:	�.
batch_normalization_18_15807362:	�%
dense_23_15807382:
�� 
dense_23_15807384:	�.
batch_normalization_19_15807387:	�.
batch_normalization_19_15807389:	�$
dense_24_15807403:	�<
dense_24_15807405:<$
dense_25_15807419:	�<
dense_25_15807421:<
identity

identity_1��.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall�.batch_normalization_18/StatefulPartitionedCall�.batch_normalization_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall� dense_25/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinputsdense_20_15807301dense_20_15807303*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_20_layer_call_and_return_conditional_losses_15807300�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_16_15807306batch_normalization_16_15807308*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_15807040�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_21_15807328dense_21_15807330*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_21_layer_call_and_return_conditional_losses_15807327�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_17_15807333batch_normalization_17_15807335*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_15807104�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_22_15807355dense_22_15807357*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_22_layer_call_and_return_conditional_losses_15807354�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_18_15807360batch_normalization_18_15807362*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_18_layer_call_and_return_conditional_losses_15807168�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0dense_23_15807382dense_23_15807384*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_15807381�
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0batch_normalization_19_15807387batch_normalization_19_15807389*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_19_layer_call_and_return_conditional_losses_15807232�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0dense_24_15807403dense_24_15807405*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_15807402�
 dense_25/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0dense_25_15807419dense_25_15807421*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_15807418\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
clip_by_value/MinimumMinimum)dense_25/StatefulPartitionedCall:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������<T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������<x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<b

Identity_1Identityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:���������<�
NoOpNoOp/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������<: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������<<
output_10
StatefulPartitionedCall:0���������<<
output_20
StatefulPartitionedCall:1���������<tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
hidden_layers
	batch_norm_layers

output_layer_means
output_layer_log_stds
	optimizer
loss
call
#get_action_and_log_prob_density

signatures"
_tf_keras_model
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15
!16
"17
#18
$19"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
!8
"9
#10
$11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
*trace_0
+trace_1
,trace_2
-trace_32�
6__inference_policy_network_soft_layer_call_fn_15807475
6__inference_policy_network_soft_layer_call_fn_15807953
6__inference_policy_network_soft_layer_call_fn_15808000
6__inference_policy_network_soft_layer_call_fn_15807735�
���
FullArgSpec@
args8�5
jself
jinputs

jtraining
jmasks
j
print_stds
varargs
 
varkw
 
defaults�
p 

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z*trace_0z+trace_1z,trace_2z-trace_3
�
.trace_0
/trace_1
0trace_2
1trace_32�
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15808113
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15808282
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15807794
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15807853�
���
FullArgSpec@
args8�5
jself
jinputs

jtraining
jmasks
j
print_stds
varargs
 
varkw
 
defaults�
p 

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z.trace_0z/trace_1z0trace_2z1trace_3
�B�
#__inference__wrapped_model_15807021input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
<
20
31
42
53"
trackable_list_wrapper
<
60
71
82
93"
trackable_list_wrapper
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

!kernel
"bias"
_tf_keras_layer
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
�

Fbeta_1

Gbeta_2
	Hdecay
Iiterm�m�m�m�m�m�m�m�!m�"m�#m�$m�v�v�v�v�v�v�v�v�!v�"v�#v�$v�vhat�vhat�vhat�vhat�vhat�vhat�vhat�vhat�!vhat�"vhat�#vhat�$vhat�"
	optimizer
 "
trackable_dict_wrapper
�
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_3
Ntrace_42�
__inference_call_7283
__inference_call_7397
__inference_call_7511
__inference_call_7624
__inference_call_27032�
���
FullArgSpec@
args8�5
jself
jinputs

jtraining
jmasks
j
print_stds
varargs
 
varkw
 
defaults�

 

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zJtrace_0zKtrace_1zLtrace_2zMtrace_3zNtrace_4
�
Otrace_0
Ptrace_12�
0__inference_get_action_and_log_prob_density_7826
1__inference_get_action_and_log_prob_density_27287�
���
FullArgSpec6
args.�+
jself
jstate

jtraining
j
print_stds
varargs
 
varkw
 
defaults�
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zOtrace_0zPtrace_1
,
Qserving_default"
signature_map
": 	<�2dense_20/kernel
:�2dense_20/bias
#:!
��2dense_21/kernel
:�2dense_21/bias
#:!
��2dense_22/kernel
:�2dense_22/bias
#:!
��2dense_23/kernel
:�2dense_23/bias
3:1� (2"batch_normalization_16/moving_mean
7:5� (2&batch_normalization_16/moving_variance
3:1� (2"batch_normalization_17/moving_mean
7:5� (2&batch_normalization_17/moving_variance
3:1� (2"batch_normalization_18/moving_mean
7:5� (2&batch_normalization_18/moving_variance
3:1� (2"batch_normalization_19/moving_mean
7:5� (2&batch_normalization_19/moving_variance
": 	�<2dense_24/kernel
:<2dense_24/bias
": 	�<2dense_25/kernel
:<2dense_25/bias
X
0
1
2
3
4
5
6
 7"
trackable_list_wrapper
f
20
31
42
53
64
75
86
97

8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_policy_network_soft_layer_call_fn_15807475input_1"�
���
FullArgSpec@
args8�5
jself
jinputs

jtraining
jmasks
j
print_stds
varargs
 
varkw
 
defaults�
p 

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_policy_network_soft_layer_call_fn_15807953inputs"�
���
FullArgSpec@
args8�5
jself
jinputs

jtraining
jmasks
j
print_stds
varargs
 
varkw
 
defaults�
p 

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_policy_network_soft_layer_call_fn_15808000inputs"�
���
FullArgSpec@
args8�5
jself
jinputs

jtraining
jmasks
j
print_stds
varargs
 
varkw
 
defaults�
p 

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_policy_network_soft_layer_call_fn_15807735input_1"�
���
FullArgSpec@
args8�5
jself
jinputs

jtraining
jmasks
j
print_stds
varargs
 
varkw
 
defaults�
p 

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15808113inputs"�
���
FullArgSpec@
args8�5
jself
jinputs

jtraining
jmasks
j
print_stds
varargs
 
varkw
 
defaults�
p 

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15808282inputs"�
���
FullArgSpec@
args8�5
jself
jinputs

jtraining
jmasks
j
print_stds
varargs
 
varkw
 
defaults�
p 

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15807794input_1"�
���
FullArgSpec@
args8�5
jself
jinputs

jtraining
jmasks
j
print_stds
varargs
 
varkw
 
defaults�
p 

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15807853input_1"�
���
FullArgSpec@
args8�5
jself
jinputs

jtraining
jmasks
j
print_stds
varargs
 
varkw
 
defaults�
p 

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
paxis
moving_mean
moving_variance"
_tf_keras_layer
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses
waxis
moving_mean
moving_variance"
_tf_keras_layer
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses
~axis
moving_mean
moving_variance"
_tf_keras_layer
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
moving_mean
 moving_variance"
_tf_keras_layer
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_24_layer_call_fn_15808291�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_24_layer_call_and_return_conditional_losses_15808301�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_25_layer_call_fn_15808310�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_25_layer_call_and_return_conditional_losses_15808320�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
: (2beta_1
: (2beta_2
: (2decay
:	 (2iter
�B�
__inference_call_7283inputs"�
���
FullArgSpec@
args8�5
jself
jinputs

jtraining
jmasks
j
print_stds
varargs
 
varkw
 
defaults�

 

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_call_7397inputs"�
���
FullArgSpec@
args8�5
jself
jinputs

jtraining
jmasks
j
print_stds
varargs
 
varkw
 
defaults�

 

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_call_7511inputs"�
���
FullArgSpec@
args8�5
jself
jinputs

jtraining
jmasks
j
print_stds
varargs
 
varkw
 
defaults�

 

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_call_7624inputs"�
���
FullArgSpec@
args8�5
jself
jinputs

jtraining
jmasks
j
print_stds
varargs
 
varkw
 
defaults�

 

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_call_27032inputs"�
���
FullArgSpec@
args8�5
jself
jinputs

jtraining
jmasks
j
print_stds
varargs
 
varkw
 
defaults�

 

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
0__inference_get_action_and_log_prob_density_7826state"�
���
FullArgSpec6
args.�+
jself
jstate

jtraining
j
print_stds
varargs
 
varkw
 
defaults�
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_get_action_and_log_prob_density_27287state"�
���
FullArgSpec6
args.�+
jself
jstate

jtraining
j
print_stds
varargs
 
varkw
 
defaults�
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_15807906input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_20_layer_call_fn_15808329�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_20_layer_call_and_return_conditional_losses_15808345�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_21_layer_call_fn_15808354�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_21_layer_call_and_return_conditional_losses_15808370�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_22_layer_call_fn_15808379�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_22_layer_call_and_return_conditional_losses_15808395�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_23_layer_call_fn_15808404�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_23_layer_call_and_return_conditional_losses_15808420�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_16_layer_call_fn_15808429
9__inference_batch_normalization_16_layer_call_fn_15808438�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_15808453
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_15808482�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_17_layer_call_fn_15808491
9__inference_batch_normalization_17_layer_call_fn_15808500�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_15808515
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_15808544�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_18_layer_call_fn_15808553
9__inference_batch_normalization_18_layer_call_fn_15808562�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_18_layer_call_and_return_conditional_losses_15808577
T__inference_batch_normalization_18_layer_call_and_return_conditional_losses_15808606�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_19_layer_call_fn_15808615
9__inference_batch_normalization_19_layer_call_fn_15808624�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_19_layer_call_and_return_conditional_losses_15808639
T__inference_batch_normalization_19_layer_call_and_return_conditional_losses_15808668�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
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
�B�
+__inference_dense_24_layer_call_fn_15808291inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_24_layer_call_and_return_conditional_losses_15808301inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dense_25_layer_call_fn_15808310inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_25_layer_call_and_return_conditional_losses_15808320inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dense_20_layer_call_fn_15808329inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_20_layer_call_and_return_conditional_losses_15808345inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dense_21_layer_call_fn_15808354inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_21_layer_call_and_return_conditional_losses_15808370inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dense_22_layer_call_fn_15808379inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_22_layer_call_and_return_conditional_losses_15808395inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dense_23_layer_call_fn_15808404inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_23_layer_call_and_return_conditional_losses_15808420inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_16_layer_call_fn_15808429inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_16_layer_call_fn_15808438inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_15808453inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_15808482inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_17_layer_call_fn_15808491inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_17_layer_call_fn_15808500inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_15808515inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_15808544inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_18_layer_call_fn_15808553inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_18_layer_call_fn_15808562inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_18_layer_call_and_return_conditional_losses_15808577inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_18_layer_call_and_return_conditional_losses_15808606inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_19_layer_call_fn_15808615inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_19_layer_call_fn_15808624inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_19_layer_call_and_return_conditional_losses_15808639inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_19_layer_call_and_return_conditional_losses_15808668inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
):'	<�2Adam_2/dense_20/kernel/m
#:!�2Adam_2/dense_20/bias/m
*:(
��2Adam_2/dense_21/kernel/m
#:!�2Adam_2/dense_21/bias/m
*:(
��2Adam_2/dense_22/kernel/m
#:!�2Adam_2/dense_22/bias/m
*:(
��2Adam_2/dense_23/kernel/m
#:!�2Adam_2/dense_23/bias/m
):'	�<2Adam_2/dense_24/kernel/m
": <2Adam_2/dense_24/bias/m
):'	�<2Adam_2/dense_25/kernel/m
": <2Adam_2/dense_25/bias/m
):'	<�2Adam_2/dense_20/kernel/v
#:!�2Adam_2/dense_20/bias/v
*:(
��2Adam_2/dense_21/kernel/v
#:!�2Adam_2/dense_21/bias/v
*:(
��2Adam_2/dense_22/kernel/v
#:!�2Adam_2/dense_22/bias/v
*:(
��2Adam_2/dense_23/kernel/v
#:!�2Adam_2/dense_23/bias/v
):'	�<2Adam_2/dense_24/kernel/v
": <2Adam_2/dense_24/bias/v
):'	�<2Adam_2/dense_25/kernel/v
": <2Adam_2/dense_25/bias/v
,:*	<�2Adam_2/dense_20/kernel/vhat
&:$�2Adam_2/dense_20/bias/vhat
-:+
��2Adam_2/dense_21/kernel/vhat
&:$�2Adam_2/dense_21/bias/vhat
-:+
��2Adam_2/dense_22/kernel/vhat
&:$�2Adam_2/dense_22/bias/vhat
-:+
��2Adam_2/dense_23/kernel/vhat
&:$�2Adam_2/dense_23/bias/vhat
,:*	�<2Adam_2/dense_24/kernel/vhat
%:#<2Adam_2/dense_24/bias/vhat
,:*	�<2Adam_2/dense_25/kernel/vhat
%:#<2Adam_2/dense_25/bias/vhat�
#__inference__wrapped_model_15807021� !"#$0�-
&�#
!�
input_1���������<
� "c�`
.
output_1"�
output_1���������<
.
output_2"�
output_2���������<�
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_15808453b4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
T__inference_batch_normalization_16_layer_call_and_return_conditional_losses_15808482b4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
9__inference_batch_normalization_16_layer_call_fn_15808429U4�1
*�'
!�
inputs����������
p 
� "������������
9__inference_batch_normalization_16_layer_call_fn_15808438U4�1
*�'
!�
inputs����������
p
� "������������
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_15808515b4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
T__inference_batch_normalization_17_layer_call_and_return_conditional_losses_15808544b4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
9__inference_batch_normalization_17_layer_call_fn_15808491U4�1
*�'
!�
inputs����������
p 
� "������������
9__inference_batch_normalization_17_layer_call_fn_15808500U4�1
*�'
!�
inputs����������
p
� "������������
T__inference_batch_normalization_18_layer_call_and_return_conditional_losses_15808577b4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
T__inference_batch_normalization_18_layer_call_and_return_conditional_losses_15808606b4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
9__inference_batch_normalization_18_layer_call_fn_15808553U4�1
*�'
!�
inputs����������
p 
� "������������
9__inference_batch_normalization_18_layer_call_fn_15808562U4�1
*�'
!�
inputs����������
p
� "������������
T__inference_batch_normalization_19_layer_call_and_return_conditional_losses_15808639b 4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
T__inference_batch_normalization_19_layer_call_and_return_conditional_losses_15808668b 4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
9__inference_batch_normalization_19_layer_call_fn_15808615U 4�1
*�'
!�
inputs����������
p 
� "������������
9__inference_batch_normalization_19_layer_call_fn_15808624U 4�1
*�'
!�
inputs����������
p
� "������������
__inference_call_27032z !"#$3�0
)�&
�
inputs	�<
p

 
p 
� "-�*
�
0	�<
�
1	�<�
__inference_call_7283w !"#$2�/
(�%
�
inputs<

 

 
p 
� "+�(
�
0<
�
1<�
__inference_call_7397w !"#$2�/
(�%
�
inputs<

 

 
p 
� "+�(
�
0<
�
1<�
__inference_call_7511w !"#$2�/
(�%
�
inputs<
p 

 
p 
� "+�(
�
0<
�
1<�
__inference_call_7624� !"#$;�8
1�.
 �
inputs���������<
p 

 
p 
� "=�:
�
0���������<
�
1���������<�
F__inference_dense_20_layer_call_and_return_conditional_losses_15808345]/�,
%�"
 �
inputs���������<
� "&�#
�
0����������
� 
+__inference_dense_20_layer_call_fn_15808329P/�,
%�"
 �
inputs���������<
� "������������
F__inference_dense_21_layer_call_and_return_conditional_losses_15808370^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
+__inference_dense_21_layer_call_fn_15808354Q0�-
&�#
!�
inputs����������
� "������������
F__inference_dense_22_layer_call_and_return_conditional_losses_15808395^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
+__inference_dense_22_layer_call_fn_15808379Q0�-
&�#
!�
inputs����������
� "������������
F__inference_dense_23_layer_call_and_return_conditional_losses_15808420^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
+__inference_dense_23_layer_call_fn_15808404Q0�-
&�#
!�
inputs����������
� "������������
F__inference_dense_24_layer_call_and_return_conditional_losses_15808301]!"0�-
&�#
!�
inputs����������
� "%�"
�
0���������<
� 
+__inference_dense_24_layer_call_fn_15808291P!"0�-
&�#
!�
inputs����������
� "����������<�
F__inference_dense_25_layer_call_and_return_conditional_losses_15808320]#$0�-
&�#
!�
inputs����������
� "%�"
�
0���������<
� 
+__inference_dense_25_layer_call_fn_15808310P#$0�-
&�#
!�
inputs����������
� "����������<�
1__inference_get_action_and_log_prob_density_27287u !"#$.�+
$�!
�
state	�<
p
p 
� "-�*
�
0	�<
�
1	�<�
0__inference_get_action_and_log_prob_density_7826n !"#$)�&
�
�
state<
p 
p 
� "+�(
�
0<
�
1<�
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15807794� !"#$<�9
2�/
!�
input_1���������<
p 

 
p 
� "K�H
A�>
�
0/0���������<
�
0/1���������<
� �
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15807853� !"#$<�9
2�/
!�
input_1���������<
p

 
p 
� "K�H
A�>
�
0/0���������<
�
0/1���������<
� �
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15808113� !"#$;�8
1�.
 �
inputs���������<
p 

 
p 
� "K�H
A�>
�
0/0���������<
�
0/1���������<
� �
Q__inference_policy_network_soft_layer_call_and_return_conditional_losses_15808282� !"#$;�8
1�.
 �
inputs���������<
p

 
p 
� "K�H
A�>
�
0/0���������<
�
0/1���������<
� �
6__inference_policy_network_soft_layer_call_fn_15807475� !"#$<�9
2�/
!�
input_1���������<
p 

 
p 
� "=�:
�
0���������<
�
1���������<�
6__inference_policy_network_soft_layer_call_fn_15807735� !"#$<�9
2�/
!�
input_1���������<
p

 
p 
� "=�:
�
0���������<
�
1���������<�
6__inference_policy_network_soft_layer_call_fn_15807953� !"#$;�8
1�.
 �
inputs���������<
p 

 
p 
� "=�:
�
0���������<
�
1���������<�
6__inference_policy_network_soft_layer_call_fn_15808000� !"#$;�8
1�.
 �
inputs���������<
p

 
p 
� "=�:
�
0���������<
�
1���������<�
&__inference_signature_wrapper_15807906� !"#$;�8
� 
1�.
,
input_1!�
input_1���������<"c�`
.
output_1"�
output_1���������<
.
output_2"�
output_2���������<
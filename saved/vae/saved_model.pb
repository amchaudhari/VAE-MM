£
ô
É

B
AssignVariableOp
resource
value"dtype"
dtypetype
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

encoder/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameencoder/conv2d_2/kernel

+encoder/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpencoder/conv2d_2/kernel*&
_output_shapes
: *
dtype0

encoder/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameencoder/conv2d_2/bias
{
)encoder/conv2d_2/bias/Read/ReadVariableOpReadVariableOpencoder/conv2d_2/bias*
_output_shapes
: *
dtype0

encoder/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameencoder/conv2d_3/kernel

+encoder/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpencoder/conv2d_3/kernel*&
_output_shapes
: @*
dtype0

encoder/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameencoder/conv2d_3/bias
{
)encoder/conv2d_3/bias/Read/ReadVariableOpReadVariableOpencoder/conv2d_3/bias*
_output_shapes
:@*
dtype0

encoder/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	À*'
shared_nameencoder/dense_5/kernel

*encoder/dense_5/kernel/Read/ReadVariableOpReadVariableOpencoder/dense_5/kernel*
_output_shapes
:	À*
dtype0

encoder/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameencoder/dense_5/bias
y
(encoder/dense_5/bias/Read/ReadVariableOpReadVariableOpencoder/dense_5/bias*
_output_shapes
:*
dtype0

encoder/z_mu/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameencoder/z_mu/kernel
{
'encoder/z_mu/kernel/Read/ReadVariableOpReadVariableOpencoder/z_mu/kernel*
_output_shapes

:*
dtype0
z
encoder/z_mu/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameencoder/z_mu/bias
s
%encoder/z_mu/bias/Read/ReadVariableOpReadVariableOpencoder/z_mu/bias*
_output_shapes
:*
dtype0

encoder/z_log_sigma/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameencoder/z_log_sigma/kernel

.encoder/z_log_sigma/kernel/Read/ReadVariableOpReadVariableOpencoder/z_log_sigma/kernel*
_output_shapes

:*
dtype0

encoder/z_log_sigma/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameencoder/z_log_sigma/bias

,encoder/z_log_sigma/bias/Read/ReadVariableOpReadVariableOpencoder/z_log_sigma/bias*
_output_shapes
:*
dtype0

decoder/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
À*'
shared_namedecoder/dense_6/kernel

*decoder/dense_6/kernel/Read/ReadVariableOpReadVariableOpdecoder/dense_6/kernel*
_output_shapes
:	
À*
dtype0

decoder/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*%
shared_namedecoder/dense_6/bias
z
(decoder/dense_6/bias/Read/ReadVariableOpReadVariableOpdecoder/dense_6/bias*
_output_shapes	
:À*
dtype0
¦
!decoder/conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!decoder/conv2d_transpose_3/kernel

5decoder/conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_3/kernel*&
_output_shapes
:@@*
dtype0

decoder/conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!decoder/conv2d_transpose_3/bias

3decoder/conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_3/bias*
_output_shapes
:@*
dtype0
¦
!decoder/conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!decoder/conv2d_transpose_4/kernel

5decoder/conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_4/kernel*&
_output_shapes
: @*
dtype0

decoder/conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!decoder/conv2d_transpose_4/bias

3decoder/conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_4/bias*
_output_shapes
: *
dtype0
¦
!decoder/conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!decoder/conv2d_transpose_5/kernel

5decoder/conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_5/kernel*&
_output_shapes
: *
dtype0

decoder/conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!decoder/conv2d_transpose_5/bias

3decoder/conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_5/bias*
_output_shapes
:*
dtype0

regressor/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
d*)
shared_nameregressor/dense_7/kernel

,regressor/dense_7/kernel/Read/ReadVariableOpReadVariableOpregressor/dense_7/kernel*
_output_shapes

:
d*
dtype0

regressor/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*'
shared_nameregressor/dense_7/bias
}
*regressor/dense_7/bias/Read/ReadVariableOpReadVariableOpregressor/dense_7/bias*
_output_shapes
:d*
dtype0

regressor/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*)
shared_nameregressor/dense_8/kernel

,regressor/dense_8/kernel/Read/ReadVariableOpReadVariableOpregressor/dense_8/kernel*
_output_shapes

:d*
dtype0

regressor/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameregressor/dense_8/bias
}
*regressor/dense_8/bias/Read/ReadVariableOpReadVariableOpregressor/dense_8/bias*
_output_shapes
:*
dtype0

regressor/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameregressor/dense_9/kernel

,regressor/dense_9/kernel/Read/ReadVariableOpReadVariableOpregressor/dense_9/kernel*
_output_shapes

:*
dtype0

regressor/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameregressor/dense_9/bias
}
*regressor/dense_9/bias/Read/ReadVariableOpReadVariableOpregressor/dense_9/bias*
_output_shapes
:*
dtype0
 
Adam/encoder/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/encoder/conv2d_2/kernel/m

2Adam/encoder/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encoder/conv2d_2/kernel/m*&
_output_shapes
: *
dtype0

Adam/encoder/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/encoder/conv2d_2/bias/m

0Adam/encoder/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/encoder/conv2d_2/bias/m*
_output_shapes
: *
dtype0
 
Adam/encoder/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*/
shared_name Adam/encoder/conv2d_3/kernel/m

2Adam/encoder/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encoder/conv2d_3/kernel/m*&
_output_shapes
: @*
dtype0

Adam/encoder/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/encoder/conv2d_3/bias/m

0Adam/encoder/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/encoder/conv2d_3/bias/m*
_output_shapes
:@*
dtype0

Adam/encoder/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	À*.
shared_nameAdam/encoder/dense_5/kernel/m

1Adam/encoder/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encoder/dense_5/kernel/m*
_output_shapes
:	À*
dtype0

Adam/encoder/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/encoder/dense_5/bias/m

/Adam/encoder/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/encoder/dense_5/bias/m*
_output_shapes
:*
dtype0

Adam/encoder/z_mu/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/encoder/z_mu/kernel/m

.Adam/encoder/z_mu/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encoder/z_mu/kernel/m*
_output_shapes

:*
dtype0

Adam/encoder/z_mu/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/encoder/z_mu/bias/m

,Adam/encoder/z_mu/bias/m/Read/ReadVariableOpReadVariableOpAdam/encoder/z_mu/bias/m*
_output_shapes
:*
dtype0

!Adam/encoder/z_log_sigma/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/encoder/z_log_sigma/kernel/m

5Adam/encoder/z_log_sigma/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/encoder/z_log_sigma/kernel/m*
_output_shapes

:*
dtype0

Adam/encoder/z_log_sigma/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/encoder/z_log_sigma/bias/m

3Adam/encoder/z_log_sigma/bias/m/Read/ReadVariableOpReadVariableOpAdam/encoder/z_log_sigma/bias/m*
_output_shapes
:*
dtype0

Adam/decoder/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
À*.
shared_nameAdam/decoder/dense_6/kernel/m

1Adam/decoder/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoder/dense_6/kernel/m*
_output_shapes
:	
À*
dtype0

Adam/decoder/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*,
shared_nameAdam/decoder/dense_6/bias/m

/Adam/decoder/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/decoder/dense_6/bias/m*
_output_shapes	
:À*
dtype0
´
(Adam/decoder/conv2d_transpose_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*9
shared_name*(Adam/decoder/conv2d_transpose_3/kernel/m
­
<Adam/decoder/conv2d_transpose_3/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/decoder/conv2d_transpose_3/kernel/m*&
_output_shapes
:@@*
dtype0
¤
&Adam/decoder/conv2d_transpose_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&Adam/decoder/conv2d_transpose_3/bias/m

:Adam/decoder/conv2d_transpose_3/bias/m/Read/ReadVariableOpReadVariableOp&Adam/decoder/conv2d_transpose_3/bias/m*
_output_shapes
:@*
dtype0
´
(Adam/decoder/conv2d_transpose_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*9
shared_name*(Adam/decoder/conv2d_transpose_4/kernel/m
­
<Adam/decoder/conv2d_transpose_4/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/decoder/conv2d_transpose_4/kernel/m*&
_output_shapes
: @*
dtype0
¤
&Adam/decoder/conv2d_transpose_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/decoder/conv2d_transpose_4/bias/m

:Adam/decoder/conv2d_transpose_4/bias/m/Read/ReadVariableOpReadVariableOp&Adam/decoder/conv2d_transpose_4/bias/m*
_output_shapes
: *
dtype0
´
(Adam/decoder/conv2d_transpose_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(Adam/decoder/conv2d_transpose_5/kernel/m
­
<Adam/decoder/conv2d_transpose_5/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/decoder/conv2d_transpose_5/kernel/m*&
_output_shapes
: *
dtype0
¤
&Adam/decoder/conv2d_transpose_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/decoder/conv2d_transpose_5/bias/m

:Adam/decoder/conv2d_transpose_5/bias/m/Read/ReadVariableOpReadVariableOp&Adam/decoder/conv2d_transpose_5/bias/m*
_output_shapes
:*
dtype0

Adam/regressor/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
d*0
shared_name!Adam/regressor/dense_7/kernel/m

3Adam/regressor/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/regressor/dense_7/kernel/m*
_output_shapes

:
d*
dtype0

Adam/regressor/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*.
shared_nameAdam/regressor/dense_7/bias/m

1Adam/regressor/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/regressor/dense_7/bias/m*
_output_shapes
:d*
dtype0

Adam/regressor/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*0
shared_name!Adam/regressor/dense_8/kernel/m

3Adam/regressor/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/regressor/dense_8/kernel/m*
_output_shapes

:d*
dtype0

Adam/regressor/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/regressor/dense_8/bias/m

1Adam/regressor/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/regressor/dense_8/bias/m*
_output_shapes
:*
dtype0

Adam/regressor/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/regressor/dense_9/kernel/m

3Adam/regressor/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/regressor/dense_9/kernel/m*
_output_shapes

:*
dtype0

Adam/regressor/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/regressor/dense_9/bias/m

1Adam/regressor/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/regressor/dense_9/bias/m*
_output_shapes
:*
dtype0
 
Adam/encoder/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/encoder/conv2d_2/kernel/v

2Adam/encoder/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encoder/conv2d_2/kernel/v*&
_output_shapes
: *
dtype0

Adam/encoder/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/encoder/conv2d_2/bias/v

0Adam/encoder/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/encoder/conv2d_2/bias/v*
_output_shapes
: *
dtype0
 
Adam/encoder/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*/
shared_name Adam/encoder/conv2d_3/kernel/v

2Adam/encoder/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encoder/conv2d_3/kernel/v*&
_output_shapes
: @*
dtype0

Adam/encoder/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/encoder/conv2d_3/bias/v

0Adam/encoder/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/encoder/conv2d_3/bias/v*
_output_shapes
:@*
dtype0

Adam/encoder/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	À*.
shared_nameAdam/encoder/dense_5/kernel/v

1Adam/encoder/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encoder/dense_5/kernel/v*
_output_shapes
:	À*
dtype0

Adam/encoder/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/encoder/dense_5/bias/v

/Adam/encoder/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/encoder/dense_5/bias/v*
_output_shapes
:*
dtype0

Adam/encoder/z_mu/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/encoder/z_mu/kernel/v

.Adam/encoder/z_mu/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encoder/z_mu/kernel/v*
_output_shapes

:*
dtype0

Adam/encoder/z_mu/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/encoder/z_mu/bias/v

,Adam/encoder/z_mu/bias/v/Read/ReadVariableOpReadVariableOpAdam/encoder/z_mu/bias/v*
_output_shapes
:*
dtype0

!Adam/encoder/z_log_sigma/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/encoder/z_log_sigma/kernel/v

5Adam/encoder/z_log_sigma/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/encoder/z_log_sigma/kernel/v*
_output_shapes

:*
dtype0

Adam/encoder/z_log_sigma/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/encoder/z_log_sigma/bias/v

3Adam/encoder/z_log_sigma/bias/v/Read/ReadVariableOpReadVariableOpAdam/encoder/z_log_sigma/bias/v*
_output_shapes
:*
dtype0

Adam/decoder/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
À*.
shared_nameAdam/decoder/dense_6/kernel/v

1Adam/decoder/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoder/dense_6/kernel/v*
_output_shapes
:	
À*
dtype0

Adam/decoder/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*,
shared_nameAdam/decoder/dense_6/bias/v

/Adam/decoder/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/decoder/dense_6/bias/v*
_output_shapes	
:À*
dtype0
´
(Adam/decoder/conv2d_transpose_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*9
shared_name*(Adam/decoder/conv2d_transpose_3/kernel/v
­
<Adam/decoder/conv2d_transpose_3/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/decoder/conv2d_transpose_3/kernel/v*&
_output_shapes
:@@*
dtype0
¤
&Adam/decoder/conv2d_transpose_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&Adam/decoder/conv2d_transpose_3/bias/v

:Adam/decoder/conv2d_transpose_3/bias/v/Read/ReadVariableOpReadVariableOp&Adam/decoder/conv2d_transpose_3/bias/v*
_output_shapes
:@*
dtype0
´
(Adam/decoder/conv2d_transpose_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*9
shared_name*(Adam/decoder/conv2d_transpose_4/kernel/v
­
<Adam/decoder/conv2d_transpose_4/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/decoder/conv2d_transpose_4/kernel/v*&
_output_shapes
: @*
dtype0
¤
&Adam/decoder/conv2d_transpose_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/decoder/conv2d_transpose_4/bias/v

:Adam/decoder/conv2d_transpose_4/bias/v/Read/ReadVariableOpReadVariableOp&Adam/decoder/conv2d_transpose_4/bias/v*
_output_shapes
: *
dtype0
´
(Adam/decoder/conv2d_transpose_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(Adam/decoder/conv2d_transpose_5/kernel/v
­
<Adam/decoder/conv2d_transpose_5/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/decoder/conv2d_transpose_5/kernel/v*&
_output_shapes
: *
dtype0
¤
&Adam/decoder/conv2d_transpose_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/decoder/conv2d_transpose_5/bias/v

:Adam/decoder/conv2d_transpose_5/bias/v/Read/ReadVariableOpReadVariableOp&Adam/decoder/conv2d_transpose_5/bias/v*
_output_shapes
:*
dtype0

Adam/regressor/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
d*0
shared_name!Adam/regressor/dense_7/kernel/v

3Adam/regressor/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/regressor/dense_7/kernel/v*
_output_shapes

:
d*
dtype0

Adam/regressor/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*.
shared_nameAdam/regressor/dense_7/bias/v

1Adam/regressor/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/regressor/dense_7/bias/v*
_output_shapes
:d*
dtype0

Adam/regressor/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*0
shared_name!Adam/regressor/dense_8/kernel/v

3Adam/regressor/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/regressor/dense_8/kernel/v*
_output_shapes

:d*
dtype0

Adam/regressor/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/regressor/dense_8/bias/v

1Adam/regressor/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/regressor/dense_8/bias/v*
_output_shapes
:*
dtype0

Adam/regressor/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/regressor/dense_9/kernel/v

3Adam/regressor/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/regressor/dense_9/kernel/v*
_output_shapes

:*
dtype0

Adam/regressor/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/regressor/dense_9/bias/v

1Adam/regressor/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/regressor/dense_9/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
¤}
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ß|
valueÕ|BÒ| BË|
a
encoder
decoder
	regressor
	optimizer
loss
	keras_api

signatures
¤
	conv1
		conv2

flat

dense1
dense21
dense22

sample
regularization_losses
	variables
trainable_variables
	keras_api

	dense
reshape
	conv1
	conv2
	conv3
regularization_losses
	variables
trainable_variables
	keras_api
v

dense1

dense2

dense3
regularization_losses
 	variables
!trainable_variables
"	keras_api
¤
#iter

$beta_1

%beta_2
	&decay
'learning_rate(mÖ)m×.mØ/mÙ8mÚ9mÛ>mÜ?mÝDmÞEmßSmàTmá]mâ^mãcmädmåimæjmçtmèumézmê{më	mì	mí(vî)vï.vð/vñ8vò9vó>vô?võDvöEv÷SvøTvù]vú^vûcvüdvýivþjvÿtvuvzv{v	v	v
 
 
 
h

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
h

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
R
4	variables
5regularization_losses
6trainable_variables
7	keras_api
h

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
h

>kernel
?bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
h

Dkernel
Ebias
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
R
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
 
F
(0
)1
.2
/3
84
95
>6
?7
D8
E9
F
(0
)1
.2
/3
84
95
>6
?7
D8
E9
­
Nlayer_metrics

Olayers
Player_regularization_losses
Qmetrics
regularization_losses
	variables
Rnon_trainable_variables
trainable_variables
h

Skernel
Tbias
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
R
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
h

]kernel
^bias
_	variables
`regularization_losses
atrainable_variables
b	keras_api
h

ckernel
dbias
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
h

ikernel
jbias
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
 
8
S0
T1
]2
^3
c4
d5
i6
j7
8
S0
T1
]2
^3
c4
d5
i6
j7
­
olayer_metrics

players
qlayer_regularization_losses
rmetrics
regularization_losses
	variables
snon_trainable_variables
trainable_variables
h

tkernel
ubias
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
h

zkernel
{bias
|	variables
}regularization_losses
~trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
 
,
t0
u1
z2
{3
4
5
,
t0
u1
z2
{3
4
5
²
layer_metrics
layers
 layer_regularization_losses
metrics
regularization_losses
 	variables
non_trainable_variables
!trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEencoder/conv2d_2/kernel/encoder/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEencoder/conv2d_2/bias-encoder/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
²
*	variables
layer_metrics
 layer_regularization_losses
metrics
+regularization_losses
layers
non_trainable_variables
,trainable_variables
\Z
VARIABLE_VALUEencoder/conv2d_3/kernel/encoder/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEencoder/conv2d_3/bias-encoder/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1
²
0	variables
layer_metrics
 layer_regularization_losses
metrics
1regularization_losses
layers
non_trainable_variables
2trainable_variables
 
 
 
²
4	variables
layer_metrics
 layer_regularization_losses
metrics
5regularization_losses
layers
non_trainable_variables
6trainable_variables
\Z
VARIABLE_VALUEencoder/dense_5/kernel0encoder/dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEencoder/dense_5/bias.encoder/dense1/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91
 

80
91
²
:	variables
layer_metrics
 layer_regularization_losses
metrics
;regularization_losses
layers
non_trainable_variables
<trainable_variables
ZX
VARIABLE_VALUEencoder/z_mu/kernel1encoder/dense21/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEencoder/z_mu/bias/encoder/dense21/bias/.ATTRIBUTES/VARIABLE_VALUE

>0
?1
 

>0
?1
²
@	variables
layer_metrics
  layer_regularization_losses
¡metrics
Aregularization_losses
¢layers
£non_trainable_variables
Btrainable_variables
a_
VARIABLE_VALUEencoder/z_log_sigma/kernel1encoder/dense22/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEencoder/z_log_sigma/bias/encoder/dense22/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1
 

D0
E1
²
F	variables
¤layer_metrics
 ¥layer_regularization_losses
¦metrics
Gregularization_losses
§layers
¨non_trainable_variables
Htrainable_variables
 
 
 
²
J	variables
©layer_metrics
 ªlayer_regularization_losses
«metrics
Kregularization_losses
¬layers
­non_trainable_variables
Ltrainable_variables
 
1
0
	1

2
3
4
5
6
 
 
 
[Y
VARIABLE_VALUEdecoder/dense_6/kernel/decoder/dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdecoder/dense_6/bias-decoder/dense/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1
 

S0
T1
²
U	variables
®layer_metrics
 ¯layer_regularization_losses
°metrics
Vregularization_losses
±layers
²non_trainable_variables
Wtrainable_variables
 
 
 
²
Y	variables
³layer_metrics
 ´layer_regularization_losses
µmetrics
Zregularization_losses
¶layers
·non_trainable_variables
[trainable_variables
fd
VARIABLE_VALUE!decoder/conv2d_transpose_3/kernel/decoder/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEdecoder/conv2d_transpose_3/bias-decoder/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

]0
^1
 

]0
^1
²
_	variables
¸layer_metrics
 ¹layer_regularization_losses
ºmetrics
`regularization_losses
»layers
¼non_trainable_variables
atrainable_variables
fd
VARIABLE_VALUE!decoder/conv2d_transpose_4/kernel/decoder/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEdecoder/conv2d_transpose_4/bias-decoder/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

c0
d1
 

c0
d1
²
e	variables
½layer_metrics
 ¾layer_regularization_losses
¿metrics
fregularization_losses
Àlayers
Ánon_trainable_variables
gtrainable_variables
fd
VARIABLE_VALUE!decoder/conv2d_transpose_5/kernel/decoder/conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEdecoder/conv2d_transpose_5/bias-decoder/conv3/bias/.ATTRIBUTES/VARIABLE_VALUE

i0
j1
 

i0
j1
²
k	variables
Âlayer_metrics
 Ãlayer_regularization_losses
Ämetrics
lregularization_losses
Ålayers
Ænon_trainable_variables
mtrainable_variables
 
#
0
1
2
3
4
 
 
 
`^
VARIABLE_VALUEregressor/dense_7/kernel2regressor/dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEregressor/dense_7/bias0regressor/dense1/bias/.ATTRIBUTES/VARIABLE_VALUE

t0
u1
 

t0
u1
²
v	variables
Çlayer_metrics
 Èlayer_regularization_losses
Émetrics
wregularization_losses
Êlayers
Ënon_trainable_variables
xtrainable_variables
`^
VARIABLE_VALUEregressor/dense_8/kernel2regressor/dense2/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEregressor/dense_8/bias0regressor/dense2/bias/.ATTRIBUTES/VARIABLE_VALUE

z0
{1
 

z0
{1
²
|	variables
Ìlayer_metrics
 Ílayer_regularization_losses
Îmetrics
}regularization_losses
Ïlayers
Ðnon_trainable_variables
~trainable_variables
`^
VARIABLE_VALUEregressor/dense_9/kernel2regressor/dense3/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEregressor/dense_9/bias0regressor/dense3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
µ
	variables
Ñlayer_metrics
 Òlayer_regularization_losses
Ómetrics
regularization_losses
Ôlayers
Õnon_trainable_variables
trainable_variables
 

0
1
2
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
}
VARIABLE_VALUEAdam/encoder/conv2d_2/kernel/mKencoder/conv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/encoder/conv2d_2/bias/mIencoder/conv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/encoder/conv2d_3/kernel/mKencoder/conv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/encoder/conv2d_3/bias/mIencoder/conv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/encoder/dense_5/kernel/mLencoder/dense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/encoder/dense_5/bias/mJencoder/dense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/encoder/z_mu/kernel/mMencoder/dense21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/encoder/z_mu/bias/mKencoder/dense21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/encoder/z_log_sigma/kernel/mMencoder/dense22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/encoder/z_log_sigma/bias/mKencoder/dense22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/decoder/dense_6/kernel/mKdecoder/dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/decoder/dense_6/bias/mIdecoder/dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(Adam/decoder/conv2d_transpose_3/kernel/mKdecoder/conv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&Adam/decoder/conv2d_transpose_3/bias/mIdecoder/conv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(Adam/decoder/conv2d_transpose_4/kernel/mKdecoder/conv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&Adam/decoder/conv2d_transpose_4/bias/mIdecoder/conv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(Adam/decoder/conv2d_transpose_5/kernel/mKdecoder/conv3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&Adam/decoder/conv2d_transpose_5/bias/mIdecoder/conv3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/regressor/dense_7/kernel/mNregressor/dense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/regressor/dense_7/bias/mLregressor/dense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/regressor/dense_8/kernel/mNregressor/dense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/regressor/dense_8/bias/mLregressor/dense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/regressor/dense_9/kernel/mNregressor/dense3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/regressor/dense_9/bias/mLregressor/dense3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/encoder/conv2d_2/kernel/vKencoder/conv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/encoder/conv2d_2/bias/vIencoder/conv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/encoder/conv2d_3/kernel/vKencoder/conv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/encoder/conv2d_3/bias/vIencoder/conv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/encoder/dense_5/kernel/vLencoder/dense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/encoder/dense_5/bias/vJencoder/dense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/encoder/z_mu/kernel/vMencoder/dense21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/encoder/z_mu/bias/vKencoder/dense21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/encoder/z_log_sigma/kernel/vMencoder/dense22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/encoder/z_log_sigma/bias/vKencoder/dense22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/decoder/dense_6/kernel/vKdecoder/dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/decoder/dense_6/bias/vIdecoder/dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(Adam/decoder/conv2d_transpose_3/kernel/vKdecoder/conv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&Adam/decoder/conv2d_transpose_3/bias/vIdecoder/conv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(Adam/decoder/conv2d_transpose_4/kernel/vKdecoder/conv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&Adam/decoder/conv2d_transpose_4/bias/vIdecoder/conv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(Adam/decoder/conv2d_transpose_5/kernel/vKdecoder/conv3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&Adam/decoder/conv2d_transpose_5/bias/vIdecoder/conv3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/regressor/dense_7/kernel/vNregressor/dense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/regressor/dense_7/bias/vLregressor/dense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/regressor/dense_8/kernel/vNregressor/dense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/regressor/dense_8/bias/vLregressor/dense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/regressor/dense_9/kernel/vNregressor/dense3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/regressor/dense_9/bias/vLregressor/dense3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ó 
StatefulPartitionedCallStatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+encoder/conv2d_2/kernel/Read/ReadVariableOp)encoder/conv2d_2/bias/Read/ReadVariableOp+encoder/conv2d_3/kernel/Read/ReadVariableOp)encoder/conv2d_3/bias/Read/ReadVariableOp*encoder/dense_5/kernel/Read/ReadVariableOp(encoder/dense_5/bias/Read/ReadVariableOp'encoder/z_mu/kernel/Read/ReadVariableOp%encoder/z_mu/bias/Read/ReadVariableOp.encoder/z_log_sigma/kernel/Read/ReadVariableOp,encoder/z_log_sigma/bias/Read/ReadVariableOp*decoder/dense_6/kernel/Read/ReadVariableOp(decoder/dense_6/bias/Read/ReadVariableOp5decoder/conv2d_transpose_3/kernel/Read/ReadVariableOp3decoder/conv2d_transpose_3/bias/Read/ReadVariableOp5decoder/conv2d_transpose_4/kernel/Read/ReadVariableOp3decoder/conv2d_transpose_4/bias/Read/ReadVariableOp5decoder/conv2d_transpose_5/kernel/Read/ReadVariableOp3decoder/conv2d_transpose_5/bias/Read/ReadVariableOp,regressor/dense_7/kernel/Read/ReadVariableOp*regressor/dense_7/bias/Read/ReadVariableOp,regressor/dense_8/kernel/Read/ReadVariableOp*regressor/dense_8/bias/Read/ReadVariableOp,regressor/dense_9/kernel/Read/ReadVariableOp*regressor/dense_9/bias/Read/ReadVariableOp2Adam/encoder/conv2d_2/kernel/m/Read/ReadVariableOp0Adam/encoder/conv2d_2/bias/m/Read/ReadVariableOp2Adam/encoder/conv2d_3/kernel/m/Read/ReadVariableOp0Adam/encoder/conv2d_3/bias/m/Read/ReadVariableOp1Adam/encoder/dense_5/kernel/m/Read/ReadVariableOp/Adam/encoder/dense_5/bias/m/Read/ReadVariableOp.Adam/encoder/z_mu/kernel/m/Read/ReadVariableOp,Adam/encoder/z_mu/bias/m/Read/ReadVariableOp5Adam/encoder/z_log_sigma/kernel/m/Read/ReadVariableOp3Adam/encoder/z_log_sigma/bias/m/Read/ReadVariableOp1Adam/decoder/dense_6/kernel/m/Read/ReadVariableOp/Adam/decoder/dense_6/bias/m/Read/ReadVariableOp<Adam/decoder/conv2d_transpose_3/kernel/m/Read/ReadVariableOp:Adam/decoder/conv2d_transpose_3/bias/m/Read/ReadVariableOp<Adam/decoder/conv2d_transpose_4/kernel/m/Read/ReadVariableOp:Adam/decoder/conv2d_transpose_4/bias/m/Read/ReadVariableOp<Adam/decoder/conv2d_transpose_5/kernel/m/Read/ReadVariableOp:Adam/decoder/conv2d_transpose_5/bias/m/Read/ReadVariableOp3Adam/regressor/dense_7/kernel/m/Read/ReadVariableOp1Adam/regressor/dense_7/bias/m/Read/ReadVariableOp3Adam/regressor/dense_8/kernel/m/Read/ReadVariableOp1Adam/regressor/dense_8/bias/m/Read/ReadVariableOp3Adam/regressor/dense_9/kernel/m/Read/ReadVariableOp1Adam/regressor/dense_9/bias/m/Read/ReadVariableOp2Adam/encoder/conv2d_2/kernel/v/Read/ReadVariableOp0Adam/encoder/conv2d_2/bias/v/Read/ReadVariableOp2Adam/encoder/conv2d_3/kernel/v/Read/ReadVariableOp0Adam/encoder/conv2d_3/bias/v/Read/ReadVariableOp1Adam/encoder/dense_5/kernel/v/Read/ReadVariableOp/Adam/encoder/dense_5/bias/v/Read/ReadVariableOp.Adam/encoder/z_mu/kernel/v/Read/ReadVariableOp,Adam/encoder/z_mu/bias/v/Read/ReadVariableOp5Adam/encoder/z_log_sigma/kernel/v/Read/ReadVariableOp3Adam/encoder/z_log_sigma/bias/v/Read/ReadVariableOp1Adam/decoder/dense_6/kernel/v/Read/ReadVariableOp/Adam/decoder/dense_6/bias/v/Read/ReadVariableOp<Adam/decoder/conv2d_transpose_3/kernel/v/Read/ReadVariableOp:Adam/decoder/conv2d_transpose_3/bias/v/Read/ReadVariableOp<Adam/decoder/conv2d_transpose_4/kernel/v/Read/ReadVariableOp:Adam/decoder/conv2d_transpose_4/bias/v/Read/ReadVariableOp<Adam/decoder/conv2d_transpose_5/kernel/v/Read/ReadVariableOp:Adam/decoder/conv2d_transpose_5/bias/v/Read/ReadVariableOp3Adam/regressor/dense_7/kernel/v/Read/ReadVariableOp1Adam/regressor/dense_7/bias/v/Read/ReadVariableOp3Adam/regressor/dense_8/kernel/v/Read/ReadVariableOp1Adam/regressor/dense_8/bias/v/Read/ReadVariableOp3Adam/regressor/dense_9/kernel/v/Read/ReadVariableOp1Adam/regressor/dense_9/bias/v/Read/ReadVariableOpConst*Z
TinS
Q2O	*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_120002
ì
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateencoder/conv2d_2/kernelencoder/conv2d_2/biasencoder/conv2d_3/kernelencoder/conv2d_3/biasencoder/dense_5/kernelencoder/dense_5/biasencoder/z_mu/kernelencoder/z_mu/biasencoder/z_log_sigma/kernelencoder/z_log_sigma/biasdecoder/dense_6/kerneldecoder/dense_6/bias!decoder/conv2d_transpose_3/kerneldecoder/conv2d_transpose_3/bias!decoder/conv2d_transpose_4/kerneldecoder/conv2d_transpose_4/bias!decoder/conv2d_transpose_5/kerneldecoder/conv2d_transpose_5/biasregressor/dense_7/kernelregressor/dense_7/biasregressor/dense_8/kernelregressor/dense_8/biasregressor/dense_9/kernelregressor/dense_9/biasAdam/encoder/conv2d_2/kernel/mAdam/encoder/conv2d_2/bias/mAdam/encoder/conv2d_3/kernel/mAdam/encoder/conv2d_3/bias/mAdam/encoder/dense_5/kernel/mAdam/encoder/dense_5/bias/mAdam/encoder/z_mu/kernel/mAdam/encoder/z_mu/bias/m!Adam/encoder/z_log_sigma/kernel/mAdam/encoder/z_log_sigma/bias/mAdam/decoder/dense_6/kernel/mAdam/decoder/dense_6/bias/m(Adam/decoder/conv2d_transpose_3/kernel/m&Adam/decoder/conv2d_transpose_3/bias/m(Adam/decoder/conv2d_transpose_4/kernel/m&Adam/decoder/conv2d_transpose_4/bias/m(Adam/decoder/conv2d_transpose_5/kernel/m&Adam/decoder/conv2d_transpose_5/bias/mAdam/regressor/dense_7/kernel/mAdam/regressor/dense_7/bias/mAdam/regressor/dense_8/kernel/mAdam/regressor/dense_8/bias/mAdam/regressor/dense_9/kernel/mAdam/regressor/dense_9/bias/mAdam/encoder/conv2d_2/kernel/vAdam/encoder/conv2d_2/bias/vAdam/encoder/conv2d_3/kernel/vAdam/encoder/conv2d_3/bias/vAdam/encoder/dense_5/kernel/vAdam/encoder/dense_5/bias/vAdam/encoder/z_mu/kernel/vAdam/encoder/z_mu/bias/v!Adam/encoder/z_log_sigma/kernel/vAdam/encoder/z_log_sigma/bias/vAdam/decoder/dense_6/kernel/vAdam/decoder/dense_6/bias/v(Adam/decoder/conv2d_transpose_3/kernel/v&Adam/decoder/conv2d_transpose_3/bias/v(Adam/decoder/conv2d_transpose_4/kernel/v&Adam/decoder/conv2d_transpose_4/bias/v(Adam/decoder/conv2d_transpose_5/kernel/v&Adam/decoder/conv2d_transpose_5/bias/vAdam/regressor/dense_7/kernel/vAdam/regressor/dense_7/bias/vAdam/regressor/dense_8/kernel/vAdam/regressor/dense_8/bias/vAdam/regressor/dense_9/kernel/vAdam/regressor/dense_9/bias/v*Y
TinR
P2N*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_120243Æº
Ú
}
(__inference_dense_7_layer_call_fn_119710

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1194212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ã
s
F__inference_sampling_1_layer_call_and_return_conditional_losses_119138

inputs
inputs_1
identityD
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_normal/stddevÆ
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02$
"random_normal/RandomStandardNormal´
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
~
)__inference_conv2d_2_layer_call_fn_119530

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1189762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í	
Ü
C__inference_dense_8_layer_call_and_return_conditional_losses_119721

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ó	
Ü
C__inference_dense_6_layer_call_and_return_conditional_losses_119662

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
À*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:À*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
è
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_119360

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
strided_slice/stack_2â
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
Reshape/shape/3º
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
	
à
G__inference_z_log_sigma_layer_call_and_return_conditional_losses_119610

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

(__inference_encoder_layer_call_fn_119180
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_1191502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ú
}
(__inference_dense_9_layer_call_fn_119750

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_1194752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_119556

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ô
z
%__inference_z_mu_layer_call_fn_119600

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_z_mu_layer_call_and_return_conditional_losses_1190702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å
È%
__inference__traced_save_120002
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_encoder_conv2d_2_kernel_read_readvariableop4
0savev2_encoder_conv2d_2_bias_read_readvariableop6
2savev2_encoder_conv2d_3_kernel_read_readvariableop4
0savev2_encoder_conv2d_3_bias_read_readvariableop5
1savev2_encoder_dense_5_kernel_read_readvariableop3
/savev2_encoder_dense_5_bias_read_readvariableop2
.savev2_encoder_z_mu_kernel_read_readvariableop0
,savev2_encoder_z_mu_bias_read_readvariableop9
5savev2_encoder_z_log_sigma_kernel_read_readvariableop7
3savev2_encoder_z_log_sigma_bias_read_readvariableop5
1savev2_decoder_dense_6_kernel_read_readvariableop3
/savev2_decoder_dense_6_bias_read_readvariableop@
<savev2_decoder_conv2d_transpose_3_kernel_read_readvariableop>
:savev2_decoder_conv2d_transpose_3_bias_read_readvariableop@
<savev2_decoder_conv2d_transpose_4_kernel_read_readvariableop>
:savev2_decoder_conv2d_transpose_4_bias_read_readvariableop@
<savev2_decoder_conv2d_transpose_5_kernel_read_readvariableop>
:savev2_decoder_conv2d_transpose_5_bias_read_readvariableop7
3savev2_regressor_dense_7_kernel_read_readvariableop5
1savev2_regressor_dense_7_bias_read_readvariableop7
3savev2_regressor_dense_8_kernel_read_readvariableop5
1savev2_regressor_dense_8_bias_read_readvariableop7
3savev2_regressor_dense_9_kernel_read_readvariableop5
1savev2_regressor_dense_9_bias_read_readvariableop=
9savev2_adam_encoder_conv2d_2_kernel_m_read_readvariableop;
7savev2_adam_encoder_conv2d_2_bias_m_read_readvariableop=
9savev2_adam_encoder_conv2d_3_kernel_m_read_readvariableop;
7savev2_adam_encoder_conv2d_3_bias_m_read_readvariableop<
8savev2_adam_encoder_dense_5_kernel_m_read_readvariableop:
6savev2_adam_encoder_dense_5_bias_m_read_readvariableop9
5savev2_adam_encoder_z_mu_kernel_m_read_readvariableop7
3savev2_adam_encoder_z_mu_bias_m_read_readvariableop@
<savev2_adam_encoder_z_log_sigma_kernel_m_read_readvariableop>
:savev2_adam_encoder_z_log_sigma_bias_m_read_readvariableop<
8savev2_adam_decoder_dense_6_kernel_m_read_readvariableop:
6savev2_adam_decoder_dense_6_bias_m_read_readvariableopG
Csavev2_adam_decoder_conv2d_transpose_3_kernel_m_read_readvariableopE
Asavev2_adam_decoder_conv2d_transpose_3_bias_m_read_readvariableopG
Csavev2_adam_decoder_conv2d_transpose_4_kernel_m_read_readvariableopE
Asavev2_adam_decoder_conv2d_transpose_4_bias_m_read_readvariableopG
Csavev2_adam_decoder_conv2d_transpose_5_kernel_m_read_readvariableopE
Asavev2_adam_decoder_conv2d_transpose_5_bias_m_read_readvariableop>
:savev2_adam_regressor_dense_7_kernel_m_read_readvariableop<
8savev2_adam_regressor_dense_7_bias_m_read_readvariableop>
:savev2_adam_regressor_dense_8_kernel_m_read_readvariableop<
8savev2_adam_regressor_dense_8_bias_m_read_readvariableop>
:savev2_adam_regressor_dense_9_kernel_m_read_readvariableop<
8savev2_adam_regressor_dense_9_bias_m_read_readvariableop=
9savev2_adam_encoder_conv2d_2_kernel_v_read_readvariableop;
7savev2_adam_encoder_conv2d_2_bias_v_read_readvariableop=
9savev2_adam_encoder_conv2d_3_kernel_v_read_readvariableop;
7savev2_adam_encoder_conv2d_3_bias_v_read_readvariableop<
8savev2_adam_encoder_dense_5_kernel_v_read_readvariableop:
6savev2_adam_encoder_dense_5_bias_v_read_readvariableop9
5savev2_adam_encoder_z_mu_kernel_v_read_readvariableop7
3savev2_adam_encoder_z_mu_bias_v_read_readvariableop@
<savev2_adam_encoder_z_log_sigma_kernel_v_read_readvariableop>
:savev2_adam_encoder_z_log_sigma_bias_v_read_readvariableop<
8savev2_adam_decoder_dense_6_kernel_v_read_readvariableop:
6savev2_adam_decoder_dense_6_bias_v_read_readvariableopG
Csavev2_adam_decoder_conv2d_transpose_3_kernel_v_read_readvariableopE
Asavev2_adam_decoder_conv2d_transpose_3_bias_v_read_readvariableopG
Csavev2_adam_decoder_conv2d_transpose_4_kernel_v_read_readvariableopE
Asavev2_adam_decoder_conv2d_transpose_4_bias_v_read_readvariableopG
Csavev2_adam_decoder_conv2d_transpose_5_kernel_v_read_readvariableopE
Asavev2_adam_decoder_conv2d_transpose_5_bias_v_read_readvariableop>
:savev2_adam_regressor_dense_7_kernel_v_read_readvariableop<
8savev2_adam_regressor_dense_7_bias_v_read_readvariableop>
:savev2_adam_regressor_dense_8_kernel_v_read_readvariableop<
8savev2_adam_regressor_dense_8_bias_v_read_readvariableop>
:savev2_adam_regressor_dense_9_kernel_v_read_readvariableop<
8savev2_adam_regressor_dense_9_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*¬(
value¢(B(NB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB/encoder/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB-encoder/conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB/encoder/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB-encoder/conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB0encoder/dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB.encoder/dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB1encoder/dense21/kernel/.ATTRIBUTES/VARIABLE_VALUEB/encoder/dense21/bias/.ATTRIBUTES/VARIABLE_VALUEB1encoder/dense22/kernel/.ATTRIBUTES/VARIABLE_VALUEB/encoder/dense22/bias/.ATTRIBUTES/VARIABLE_VALUEB/decoder/dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB-decoder/dense/bias/.ATTRIBUTES/VARIABLE_VALUEB/decoder/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB-decoder/conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB/decoder/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB-decoder/conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB/decoder/conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB-decoder/conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB2regressor/dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB0regressor/dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB2regressor/dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB0regressor/dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB2regressor/dense3/kernel/.ATTRIBUTES/VARIABLE_VALUEB0regressor/dense3/bias/.ATTRIBUTES/VARIABLE_VALUEBKencoder/conv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIencoder/conv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKencoder/conv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIencoder/conv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLencoder/dense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJencoder/dense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMencoder/dense21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKencoder/dense21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMencoder/dense22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKencoder/dense22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKdecoder/dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIdecoder/dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKdecoder/conv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIdecoder/conv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKdecoder/conv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIdecoder/conv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKdecoder/conv3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIdecoder/conv3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNregressor/dense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLregressor/dense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNregressor/dense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLregressor/dense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNregressor/dense3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLregressor/dense3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKencoder/conv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBIencoder/conv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKencoder/conv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBIencoder/conv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLencoder/dense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJencoder/dense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMencoder/dense21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKencoder/dense21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMencoder/dense22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKencoder/dense22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKdecoder/dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBIdecoder/dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKdecoder/conv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBIdecoder/conv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKdecoder/conv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBIdecoder/conv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKdecoder/conv3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBIdecoder/conv3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNregressor/dense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLregressor/dense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNregressor/dense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLregressor/dense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNregressor/dense3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLregressor/dense3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names§
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*±
value§B¤NB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices¯$
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_encoder_conv2d_2_kernel_read_readvariableop0savev2_encoder_conv2d_2_bias_read_readvariableop2savev2_encoder_conv2d_3_kernel_read_readvariableop0savev2_encoder_conv2d_3_bias_read_readvariableop1savev2_encoder_dense_5_kernel_read_readvariableop/savev2_encoder_dense_5_bias_read_readvariableop.savev2_encoder_z_mu_kernel_read_readvariableop,savev2_encoder_z_mu_bias_read_readvariableop5savev2_encoder_z_log_sigma_kernel_read_readvariableop3savev2_encoder_z_log_sigma_bias_read_readvariableop1savev2_decoder_dense_6_kernel_read_readvariableop/savev2_decoder_dense_6_bias_read_readvariableop<savev2_decoder_conv2d_transpose_3_kernel_read_readvariableop:savev2_decoder_conv2d_transpose_3_bias_read_readvariableop<savev2_decoder_conv2d_transpose_4_kernel_read_readvariableop:savev2_decoder_conv2d_transpose_4_bias_read_readvariableop<savev2_decoder_conv2d_transpose_5_kernel_read_readvariableop:savev2_decoder_conv2d_transpose_5_bias_read_readvariableop3savev2_regressor_dense_7_kernel_read_readvariableop1savev2_regressor_dense_7_bias_read_readvariableop3savev2_regressor_dense_8_kernel_read_readvariableop1savev2_regressor_dense_8_bias_read_readvariableop3savev2_regressor_dense_9_kernel_read_readvariableop1savev2_regressor_dense_9_bias_read_readvariableop9savev2_adam_encoder_conv2d_2_kernel_m_read_readvariableop7savev2_adam_encoder_conv2d_2_bias_m_read_readvariableop9savev2_adam_encoder_conv2d_3_kernel_m_read_readvariableop7savev2_adam_encoder_conv2d_3_bias_m_read_readvariableop8savev2_adam_encoder_dense_5_kernel_m_read_readvariableop6savev2_adam_encoder_dense_5_bias_m_read_readvariableop5savev2_adam_encoder_z_mu_kernel_m_read_readvariableop3savev2_adam_encoder_z_mu_bias_m_read_readvariableop<savev2_adam_encoder_z_log_sigma_kernel_m_read_readvariableop:savev2_adam_encoder_z_log_sigma_bias_m_read_readvariableop8savev2_adam_decoder_dense_6_kernel_m_read_readvariableop6savev2_adam_decoder_dense_6_bias_m_read_readvariableopCsavev2_adam_decoder_conv2d_transpose_3_kernel_m_read_readvariableopAsavev2_adam_decoder_conv2d_transpose_3_bias_m_read_readvariableopCsavev2_adam_decoder_conv2d_transpose_4_kernel_m_read_readvariableopAsavev2_adam_decoder_conv2d_transpose_4_bias_m_read_readvariableopCsavev2_adam_decoder_conv2d_transpose_5_kernel_m_read_readvariableopAsavev2_adam_decoder_conv2d_transpose_5_bias_m_read_readvariableop:savev2_adam_regressor_dense_7_kernel_m_read_readvariableop8savev2_adam_regressor_dense_7_bias_m_read_readvariableop:savev2_adam_regressor_dense_8_kernel_m_read_readvariableop8savev2_adam_regressor_dense_8_bias_m_read_readvariableop:savev2_adam_regressor_dense_9_kernel_m_read_readvariableop8savev2_adam_regressor_dense_9_bias_m_read_readvariableop9savev2_adam_encoder_conv2d_2_kernel_v_read_readvariableop7savev2_adam_encoder_conv2d_2_bias_v_read_readvariableop9savev2_adam_encoder_conv2d_3_kernel_v_read_readvariableop7savev2_adam_encoder_conv2d_3_bias_v_read_readvariableop8savev2_adam_encoder_dense_5_kernel_v_read_readvariableop6savev2_adam_encoder_dense_5_bias_v_read_readvariableop5savev2_adam_encoder_z_mu_kernel_v_read_readvariableop3savev2_adam_encoder_z_mu_bias_v_read_readvariableop<savev2_adam_encoder_z_log_sigma_kernel_v_read_readvariableop:savev2_adam_encoder_z_log_sigma_bias_v_read_readvariableop8savev2_adam_decoder_dense_6_kernel_v_read_readvariableop6savev2_adam_decoder_dense_6_bias_v_read_readvariableopCsavev2_adam_decoder_conv2d_transpose_3_kernel_v_read_readvariableopAsavev2_adam_decoder_conv2d_transpose_3_bias_v_read_readvariableopCsavev2_adam_decoder_conv2d_transpose_4_kernel_v_read_readvariableopAsavev2_adam_decoder_conv2d_transpose_4_bias_v_read_readvariableopCsavev2_adam_decoder_conv2d_transpose_5_kernel_v_read_readvariableopAsavev2_adam_decoder_conv2d_transpose_5_bias_v_read_readvariableop:savev2_adam_regressor_dense_7_kernel_v_read_readvariableop8savev2_adam_regressor_dense_7_bias_v_read_readvariableop:savev2_adam_regressor_dense_8_kernel_v_read_readvariableop8savev2_adam_regressor_dense_8_bias_v_read_readvariableop:savev2_adam_regressor_dense_9_kernel_v_read_readvariableop8savev2_adam_regressor_dense_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *\
dtypesR
P2N	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*ä
_input_shapesÒ
Ï: : : : : : : : : @:@:	À::::::	
À:À:@@:@: @: : ::
d:d:d:::: : : @:@:	À::::::	
À:À:@@:@: @: : ::
d:d:d:::: : : @:@:	À::::::	
À:À:@@:@: @: : ::
d:d:d:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 	

_output_shapes
:@:%
!

_output_shapes
:	À: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	
À:!

_output_shapes	
:À:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::$ 

_output_shapes

:
d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :, (
&
_output_shapes
: @: !

_output_shapes
:@:%"!

_output_shapes
:	À: #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::$& 

_output_shapes

:: '

_output_shapes
::%(!

_output_shapes
:	
À:!)

_output_shapes	
:À:,*(
&
_output_shapes
:@@: +

_output_shapes
:@:,,(
&
_output_shapes
: @: -

_output_shapes
: :,.(
&
_output_shapes
: : /

_output_shapes
::$0 

_output_shapes

:
d: 1

_output_shapes
:d:$2 

_output_shapes

:d: 3

_output_shapes
::$4 

_output_shapes

:: 5

_output_shapes
::,6(
&
_output_shapes
: : 7

_output_shapes
: :,8(
&
_output_shapes
: @: 9

_output_shapes
:@:%:!

_output_shapes
:	À: ;

_output_shapes
::$< 

_output_shapes

:: =

_output_shapes
::$> 

_output_shapes

:: ?

_output_shapes
::%@!

_output_shapes
:	
À:!A

_output_shapes	
:À:,B(
&
_output_shapes
:@@: C

_output_shapes
:@:,D(
&
_output_shapes
: @: E

_output_shapes
: :,F(
&
_output_shapes
: : G

_output_shapes
::$H 

_output_shapes

:
d: I

_output_shapes
:d:$J 

_output_shapes

:d: K

_output_shapes
::$L 

_output_shapes

:: M

_output_shapes
::N

_output_shapes
: 
Î

Ý
D__inference_conv2d_2_layer_call_and_return_conditional_losses_118976

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í	
Ü
C__inference_dense_9_layer_call_and_return_conditional_losses_119741

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð	
Ü
C__inference_dense_5_layer_call_and_return_conditional_losses_119044

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	À*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
	
Ù
@__inference_z_mu_layer_call_and_return_conditional_losses_119070

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

Ý
D__inference_conv2d_3_layer_call_and_return_conditional_losses_119541

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
Ù
@__inference_z_mu_layer_call_and_return_conditional_losses_119591

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
F
*__inference_reshape_1_layer_call_fn_119690

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1193602
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
í	
Ü
C__inference_dense_7_layer_call_and_return_conditional_losses_119701

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ÅË
.
"__inference__traced_restore_120243
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate.
*assignvariableop_5_encoder_conv2d_2_kernel,
(assignvariableop_6_encoder_conv2d_2_bias.
*assignvariableop_7_encoder_conv2d_3_kernel,
(assignvariableop_8_encoder_conv2d_3_bias-
)assignvariableop_9_encoder_dense_5_kernel,
(assignvariableop_10_encoder_dense_5_bias+
'assignvariableop_11_encoder_z_mu_kernel)
%assignvariableop_12_encoder_z_mu_bias2
.assignvariableop_13_encoder_z_log_sigma_kernel0
,assignvariableop_14_encoder_z_log_sigma_bias.
*assignvariableop_15_decoder_dense_6_kernel,
(assignvariableop_16_decoder_dense_6_bias9
5assignvariableop_17_decoder_conv2d_transpose_3_kernel7
3assignvariableop_18_decoder_conv2d_transpose_3_bias9
5assignvariableop_19_decoder_conv2d_transpose_4_kernel7
3assignvariableop_20_decoder_conv2d_transpose_4_bias9
5assignvariableop_21_decoder_conv2d_transpose_5_kernel7
3assignvariableop_22_decoder_conv2d_transpose_5_bias0
,assignvariableop_23_regressor_dense_7_kernel.
*assignvariableop_24_regressor_dense_7_bias0
,assignvariableop_25_regressor_dense_8_kernel.
*assignvariableop_26_regressor_dense_8_bias0
,assignvariableop_27_regressor_dense_9_kernel.
*assignvariableop_28_regressor_dense_9_bias6
2assignvariableop_29_adam_encoder_conv2d_2_kernel_m4
0assignvariableop_30_adam_encoder_conv2d_2_bias_m6
2assignvariableop_31_adam_encoder_conv2d_3_kernel_m4
0assignvariableop_32_adam_encoder_conv2d_3_bias_m5
1assignvariableop_33_adam_encoder_dense_5_kernel_m3
/assignvariableop_34_adam_encoder_dense_5_bias_m2
.assignvariableop_35_adam_encoder_z_mu_kernel_m0
,assignvariableop_36_adam_encoder_z_mu_bias_m9
5assignvariableop_37_adam_encoder_z_log_sigma_kernel_m7
3assignvariableop_38_adam_encoder_z_log_sigma_bias_m5
1assignvariableop_39_adam_decoder_dense_6_kernel_m3
/assignvariableop_40_adam_decoder_dense_6_bias_m@
<assignvariableop_41_adam_decoder_conv2d_transpose_3_kernel_m>
:assignvariableop_42_adam_decoder_conv2d_transpose_3_bias_m@
<assignvariableop_43_adam_decoder_conv2d_transpose_4_kernel_m>
:assignvariableop_44_adam_decoder_conv2d_transpose_4_bias_m@
<assignvariableop_45_adam_decoder_conv2d_transpose_5_kernel_m>
:assignvariableop_46_adam_decoder_conv2d_transpose_5_bias_m7
3assignvariableop_47_adam_regressor_dense_7_kernel_m5
1assignvariableop_48_adam_regressor_dense_7_bias_m7
3assignvariableop_49_adam_regressor_dense_8_kernel_m5
1assignvariableop_50_adam_regressor_dense_8_bias_m7
3assignvariableop_51_adam_regressor_dense_9_kernel_m5
1assignvariableop_52_adam_regressor_dense_9_bias_m6
2assignvariableop_53_adam_encoder_conv2d_2_kernel_v4
0assignvariableop_54_adam_encoder_conv2d_2_bias_v6
2assignvariableop_55_adam_encoder_conv2d_3_kernel_v4
0assignvariableop_56_adam_encoder_conv2d_3_bias_v5
1assignvariableop_57_adam_encoder_dense_5_kernel_v3
/assignvariableop_58_adam_encoder_dense_5_bias_v2
.assignvariableop_59_adam_encoder_z_mu_kernel_v0
,assignvariableop_60_adam_encoder_z_mu_bias_v9
5assignvariableop_61_adam_encoder_z_log_sigma_kernel_v7
3assignvariableop_62_adam_encoder_z_log_sigma_bias_v5
1assignvariableop_63_adam_decoder_dense_6_kernel_v3
/assignvariableop_64_adam_decoder_dense_6_bias_v@
<assignvariableop_65_adam_decoder_conv2d_transpose_3_kernel_v>
:assignvariableop_66_adam_decoder_conv2d_transpose_3_bias_v@
<assignvariableop_67_adam_decoder_conv2d_transpose_4_kernel_v>
:assignvariableop_68_adam_decoder_conv2d_transpose_4_bias_v@
<assignvariableop_69_adam_decoder_conv2d_transpose_5_kernel_v>
:assignvariableop_70_adam_decoder_conv2d_transpose_5_bias_v7
3assignvariableop_71_adam_regressor_dense_7_kernel_v5
1assignvariableop_72_adam_regressor_dense_7_bias_v7
3assignvariableop_73_adam_regressor_dense_8_kernel_v5
1assignvariableop_74_adam_regressor_dense_8_bias_v7
3assignvariableop_75_adam_regressor_dense_9_kernel_v5
1assignvariableop_76_adam_regressor_dense_9_bias_v
identity_78¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_8¢AssignVariableOp_9 )
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*¬(
value¢(B(NB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB/encoder/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB-encoder/conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB/encoder/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB-encoder/conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB0encoder/dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB.encoder/dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB1encoder/dense21/kernel/.ATTRIBUTES/VARIABLE_VALUEB/encoder/dense21/bias/.ATTRIBUTES/VARIABLE_VALUEB1encoder/dense22/kernel/.ATTRIBUTES/VARIABLE_VALUEB/encoder/dense22/bias/.ATTRIBUTES/VARIABLE_VALUEB/decoder/dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB-decoder/dense/bias/.ATTRIBUTES/VARIABLE_VALUEB/decoder/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB-decoder/conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB/decoder/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB-decoder/conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB/decoder/conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB-decoder/conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB2regressor/dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB0regressor/dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB2regressor/dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB0regressor/dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB2regressor/dense3/kernel/.ATTRIBUTES/VARIABLE_VALUEB0regressor/dense3/bias/.ATTRIBUTES/VARIABLE_VALUEBKencoder/conv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIencoder/conv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKencoder/conv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIencoder/conv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLencoder/dense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJencoder/dense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMencoder/dense21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKencoder/dense21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMencoder/dense22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKencoder/dense22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKdecoder/dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIdecoder/dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKdecoder/conv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIdecoder/conv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKdecoder/conv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIdecoder/conv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKdecoder/conv3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIdecoder/conv3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNregressor/dense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLregressor/dense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNregressor/dense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLregressor/dense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNregressor/dense3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLregressor/dense3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKencoder/conv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBIencoder/conv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKencoder/conv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBIencoder/conv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLencoder/dense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJencoder/dense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMencoder/dense21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKencoder/dense21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMencoder/dense22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKencoder/dense22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKdecoder/dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBIdecoder/dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKdecoder/conv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBIdecoder/conv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKdecoder/conv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBIdecoder/conv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKdecoder/conv3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBIdecoder/conv3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNregressor/dense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLregressor/dense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNregressor/dense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLregressor/dense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNregressor/dense3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLregressor/dense3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names­
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*±
value§B¤NB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices´
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Î
_output_shapes»
¸::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*\
dtypesR
P2N	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2£
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¢
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ª
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¯
AssignVariableOp_5AssignVariableOp*assignvariableop_5_encoder_conv2d_2_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6­
AssignVariableOp_6AssignVariableOp(assignvariableop_6_encoder_conv2d_2_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¯
AssignVariableOp_7AssignVariableOp*assignvariableop_7_encoder_conv2d_3_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8­
AssignVariableOp_8AssignVariableOp(assignvariableop_8_encoder_conv2d_3_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9®
AssignVariableOp_9AssignVariableOp)assignvariableop_9_encoder_dense_5_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10°
AssignVariableOp_10AssignVariableOp(assignvariableop_10_encoder_dense_5_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¯
AssignVariableOp_11AssignVariableOp'assignvariableop_11_encoder_z_mu_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12­
AssignVariableOp_12AssignVariableOp%assignvariableop_12_encoder_z_mu_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¶
AssignVariableOp_13AssignVariableOp.assignvariableop_13_encoder_z_log_sigma_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14´
AssignVariableOp_14AssignVariableOp,assignvariableop_14_encoder_z_log_sigma_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15²
AssignVariableOp_15AssignVariableOp*assignvariableop_15_decoder_dense_6_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16°
AssignVariableOp_16AssignVariableOp(assignvariableop_16_decoder_dense_6_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17½
AssignVariableOp_17AssignVariableOp5assignvariableop_17_decoder_conv2d_transpose_3_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18»
AssignVariableOp_18AssignVariableOp3assignvariableop_18_decoder_conv2d_transpose_3_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19½
AssignVariableOp_19AssignVariableOp5assignvariableop_19_decoder_conv2d_transpose_4_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20»
AssignVariableOp_20AssignVariableOp3assignvariableop_20_decoder_conv2d_transpose_4_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21½
AssignVariableOp_21AssignVariableOp5assignvariableop_21_decoder_conv2d_transpose_5_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22»
AssignVariableOp_22AssignVariableOp3assignvariableop_22_decoder_conv2d_transpose_5_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23´
AssignVariableOp_23AssignVariableOp,assignvariableop_23_regressor_dense_7_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24²
AssignVariableOp_24AssignVariableOp*assignvariableop_24_regressor_dense_7_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25´
AssignVariableOp_25AssignVariableOp,assignvariableop_25_regressor_dense_8_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26²
AssignVariableOp_26AssignVariableOp*assignvariableop_26_regressor_dense_8_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27´
AssignVariableOp_27AssignVariableOp,assignvariableop_27_regressor_dense_9_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28²
AssignVariableOp_28AssignVariableOp*assignvariableop_28_regressor_dense_9_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29º
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_encoder_conv2d_2_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¸
AssignVariableOp_30AssignVariableOp0assignvariableop_30_adam_encoder_conv2d_2_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31º
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adam_encoder_conv2d_3_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¸
AssignVariableOp_32AssignVariableOp0assignvariableop_32_adam_encoder_conv2d_3_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33¹
AssignVariableOp_33AssignVariableOp1assignvariableop_33_adam_encoder_dense_5_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34·
AssignVariableOp_34AssignVariableOp/assignvariableop_34_adam_encoder_dense_5_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¶
AssignVariableOp_35AssignVariableOp.assignvariableop_35_adam_encoder_z_mu_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36´
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_encoder_z_mu_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37½
AssignVariableOp_37AssignVariableOp5assignvariableop_37_adam_encoder_z_log_sigma_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38»
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_encoder_z_log_sigma_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¹
AssignVariableOp_39AssignVariableOp1assignvariableop_39_adam_decoder_dense_6_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40·
AssignVariableOp_40AssignVariableOp/assignvariableop_40_adam_decoder_dense_6_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Ä
AssignVariableOp_41AssignVariableOp<assignvariableop_41_adam_decoder_conv2d_transpose_3_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Â
AssignVariableOp_42AssignVariableOp:assignvariableop_42_adam_decoder_conv2d_transpose_3_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Ä
AssignVariableOp_43AssignVariableOp<assignvariableop_43_adam_decoder_conv2d_transpose_4_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Â
AssignVariableOp_44AssignVariableOp:assignvariableop_44_adam_decoder_conv2d_transpose_4_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Ä
AssignVariableOp_45AssignVariableOp<assignvariableop_45_adam_decoder_conv2d_transpose_5_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Â
AssignVariableOp_46AssignVariableOp:assignvariableop_46_adam_decoder_conv2d_transpose_5_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47»
AssignVariableOp_47AssignVariableOp3assignvariableop_47_adam_regressor_dense_7_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48¹
AssignVariableOp_48AssignVariableOp1assignvariableop_48_adam_regressor_dense_7_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49»
AssignVariableOp_49AssignVariableOp3assignvariableop_49_adam_regressor_dense_8_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50¹
AssignVariableOp_50AssignVariableOp1assignvariableop_50_adam_regressor_dense_8_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51»
AssignVariableOp_51AssignVariableOp3assignvariableop_51_adam_regressor_dense_9_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52¹
AssignVariableOp_52AssignVariableOp1assignvariableop_52_adam_regressor_dense_9_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53º
AssignVariableOp_53AssignVariableOp2assignvariableop_53_adam_encoder_conv2d_2_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54¸
AssignVariableOp_54AssignVariableOp0assignvariableop_54_adam_encoder_conv2d_2_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55º
AssignVariableOp_55AssignVariableOp2assignvariableop_55_adam_encoder_conv2d_3_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56¸
AssignVariableOp_56AssignVariableOp0assignvariableop_56_adam_encoder_conv2d_3_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57¹
AssignVariableOp_57AssignVariableOp1assignvariableop_57_adam_encoder_dense_5_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58·
AssignVariableOp_58AssignVariableOp/assignvariableop_58_adam_encoder_dense_5_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59¶
AssignVariableOp_59AssignVariableOp.assignvariableop_59_adam_encoder_z_mu_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60´
AssignVariableOp_60AssignVariableOp,assignvariableop_60_adam_encoder_z_mu_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61½
AssignVariableOp_61AssignVariableOp5assignvariableop_61_adam_encoder_z_log_sigma_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62»
AssignVariableOp_62AssignVariableOp3assignvariableop_62_adam_encoder_z_log_sigma_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63¹
AssignVariableOp_63AssignVariableOp1assignvariableop_63_adam_decoder_dense_6_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64·
AssignVariableOp_64AssignVariableOp/assignvariableop_64_adam_decoder_dense_6_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65Ä
AssignVariableOp_65AssignVariableOp<assignvariableop_65_adam_decoder_conv2d_transpose_3_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66Â
AssignVariableOp_66AssignVariableOp:assignvariableop_66_adam_decoder_conv2d_transpose_3_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Ä
AssignVariableOp_67AssignVariableOp<assignvariableop_67_adam_decoder_conv2d_transpose_4_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68Â
AssignVariableOp_68AssignVariableOp:assignvariableop_68_adam_decoder_conv2d_transpose_4_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69Ä
AssignVariableOp_69AssignVariableOp<assignvariableop_69_adam_decoder_conv2d_transpose_5_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Â
AssignVariableOp_70AssignVariableOp:assignvariableop_70_adam_decoder_conv2d_transpose_5_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71»
AssignVariableOp_71AssignVariableOp3assignvariableop_71_adam_regressor_dense_7_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72¹
AssignVariableOp_72AssignVariableOp1assignvariableop_72_adam_regressor_dense_7_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73»
AssignVariableOp_73AssignVariableOp3assignvariableop_73_adam_regressor_dense_8_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74¹
AssignVariableOp_74AssignVariableOp1assignvariableop_74_adam_regressor_dense_8_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75»
AssignVariableOp_75AssignVariableOp3assignvariableop_75_adam_regressor_dense_9_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76¹
AssignVariableOp_76AssignVariableOp1assignvariableop_76_adam_regressor_dense_9_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_769
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpü
Identity_77Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_77ï
Identity_78IdentityIdentity_77:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_78"#
identity_78Identity_78:output:0*Ë
_input_shapes¹
¶: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ù

3__inference_conv2d_transpose_4_layer_call_fn_119270

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1192602
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ð	
Ü
C__inference_dense_5_layer_call_and_return_conditional_losses_119572

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	À*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
ü
~
)__inference_conv2d_3_layer_call_fn_119550

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1190032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

î
C__inference_decoder_layer_call_and_return_conditional_losses_119384
input_1
dense_6_119341
dense_6_119343
conv2d_transpose_3_119368
conv2d_transpose_3_119370
conv2d_transpose_4_119373
conv2d_transpose_4_119375
conv2d_transpose_5_119378
conv2d_transpose_5_119380
identity¢*conv2d_transpose_3/StatefulPartitionedCall¢*conv2d_transpose_4/StatefulPartitionedCall¢*conv2d_transpose_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall
dense_6/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_6_119341dense_6_119343*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1193302!
dense_6/StatefulPartitionedCall
reshape_1/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1193602
reshape_1/PartitionedCallü
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_3_119368conv2d_transpose_3_119370*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1192152,
*conv2d_transpose_3/StatefulPartitionedCall
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_119373conv2d_transpose_4_119375*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1192602,
*conv2d_transpose_4/StatefulPartitionedCall
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_119378conv2d_transpose_5_119380*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_1193052,
*conv2d_transpose_5/StatefulPartitionedCallÊ
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ
::::::::2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
£
Ã
E__inference_regressor_layer_call_and_return_conditional_losses_119492
input_1
dense_7_119432
dense_7_119434
dense_8_119459
dense_8_119461
dense_9_119486
dense_9_119488
identity¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_7/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_7_119432dense_7_119434*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1194212!
dense_7/StatefulPartitionedCall±
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_119459dense_8_119461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_1194482!
dense_8/StatefulPartitionedCall±
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_119486dense_9_119488*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_1194752!
dense_9/StatefulPartitionedCallâ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ
::::::2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
ï
u
F__inference_sampling_1_layer_call_and_return_conditional_losses_119645
inputs_0
inputs_1
identityF
ShapeShapeinputs_0*
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceJ
Shape_1Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_normal/stddevÆ
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02$
"random_normal/RandomStandardNormal´
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
è
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_119685

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
strided_slice/stack_2â
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
Reshape/shape/3º
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
í	
Ü
C__inference_dense_7_layer_call_and_return_conditional_losses_119421

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
â$
û
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_119215

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOpD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stack/3
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
strided_slice_3/stack_2ì
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3³
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_transpose/ReadVariableOpð
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu»
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¦
F
*__inference_flatten_1_layer_call_fn_119561

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1190252
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ù

3__inference_conv2d_transpose_3_layer_call_fn_119225

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1192152
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
§
t
+__inference_sampling_1_layer_call_fn_119651
inputs_0
inputs_1
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sampling_1_layer_call_and_return_conditional_losses_1191382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
í	
Ü
C__inference_dense_9_layer_call_and_return_conditional_losses_119475

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

Ý
D__inference_conv2d_3_layer_call_and_return_conditional_losses_119003

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
â
¼
*__inference_regressor_layer_call_fn_119510
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_regressor_layer_call_and_return_conditional_losses_1194922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ
::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
ã

,__inference_z_log_sigma_layer_call_fn_119619

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_z_log_sigma_layer_call_and_return_conditional_losses_1190962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
}
(__inference_dense_6_layer_call_fn_119671

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1193302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ä$
û
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_119305

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOpD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stack/3
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
strided_slice_3/stack_2ì
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3³
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOpð
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
Sigmoid´
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
í	
Ü
C__inference_dense_8_layer_call_and_return_conditional_losses_119448

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ü
}
(__inference_dense_5_layer_call_fn_119581

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1190442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Ò
Ø
(__inference_decoder_layer_call_fn_119406
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_1193842
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ
::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
ó	
Ü
C__inference_dense_6_layer_call_and_return_conditional_losses_119330

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
À*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:À*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
)
£
C__inference_encoder_layer_call_and_return_conditional_losses_119150
input_1
conv2d_2_118987
conv2d_2_118989
conv2d_3_119014
conv2d_3_119016
dense_5_119055
dense_5_119057
z_mu_119081
z_mu_119083
z_log_sigma_119107
z_log_sigma_119109
identity

identity_1

identity_2¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢"sampling_1/StatefulPartitionedCall¢#z_log_sigma/StatefulPartitionedCall¢z_mu/StatefulPartitionedCall
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_2_118987conv2d_2_118989*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1189762"
 conv2d_2/StatefulPartitionedCall¿
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_119014conv2d_3_119016*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1190032"
 conv2d_3/StatefulPartitionedCallû
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1190252
flatten_1/PartitionedCall«
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_5_119055dense_5_119057*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1190442!
dense_5/StatefulPartitionedCall¢
z_mu/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0z_mu_119081z_mu_119083*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_z_mu_layer_call_and_return_conditional_losses_1190702
z_mu/StatefulPartitionedCallÅ
#z_log_sigma/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0z_log_sigma_119107z_log_sigma_119109*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_z_log_sigma_layer_call_and_return_conditional_losses_1190962%
#z_log_sigma/StatefulPartitionedCallÀ
"sampling_1/StatefulPartitionedCallStatefulPartitionedCall%z_mu/StatefulPartitionedCall:output:0,z_log_sigma/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sampling_1_layer_call_and_return_conditional_losses_1191382$
"sampling_1/StatefulPartitionedCallË
IdentityIdentity%z_mu/StatefulPartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall$^z_log_sigma/StatefulPartitionedCall^z_mu/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÖ

Identity_1Identity,z_log_sigma/StatefulPartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall$^z_log_sigma/StatefulPartitionedCall^z_mu/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1Õ

Identity_2Identity+sampling_1/StatefulPartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall$^z_log_sigma/StatefulPartitionedCall^z_mu/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2H
"sampling_1/StatefulPartitionedCall"sampling_1/StatefulPartitionedCall2J
#z_log_sigma/StatefulPartitionedCall#z_log_sigma/StatefulPartitionedCall2<
z_mu/StatefulPartitionedCallz_mu/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
½
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_119025

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ù

3__inference_conv2d_transpose_5_layer_call_fn_119315

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_1193052
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Î

Ý
D__inference_conv2d_2_layer_call_and_return_conditional_losses_119521

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
}
(__inference_dense_8_layer_call_fn_119730

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_1194482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
	
à
G__inference_z_log_sigma_layer_call_and_return_conditional_losses_119096

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â$
û
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_119260

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOpD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stack/3
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
strided_slice_3/stack_2ì
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3³
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOpð
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
Relu»
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs"±J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:ê÷

encoder
decoder
	regressor
	optimizer
loss
	keras_api

signatures"®
_tf_keras_model{"class_name": "VAE", "name": "vae_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "VAE"}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Ú
	conv1
		conv2

flat

dense1
dense21
dense22

sample
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"÷
_tf_keras_modelÝ{"class_name": "Encoder", "name": "encoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Encoder"}}
Á
	dense
reshape
	conv1
	conv2
	conv3
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"÷
_tf_keras_modelÝ{"class_name": "Decoder", "name": "decoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Decoder"}}
²

dense1

dense2

dense3
regularization_losses
 	variables
!trainable_variables
"	keras_api
__call__
+&call_and_return_all_conditional_losses"ý
_tf_keras_modelã{"class_name": "Regressor", "name": "regressor", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Regressor"}}
·
#iter

$beta_1

%beta_2
	&decay
'learning_rate(mÖ)m×.mØ/mÙ8mÚ9mÛ>mÜ?mÝDmÞEmßSmàTmá]mâ^mãcmädmåimæjmçtmèumézmê{më	mì	mí(vî)vï.vð/vñ8vò9vó>vô?võDvöEv÷SvøTvù]vú^vûcvüdvýivþjvÿtvuvzv{v	v	v"
	optimizer
 "
trackable_dict_wrapper
"
_generic_user_object
"
signature_map
ò	

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
__call__
+&call_and_return_all_conditional_losses"Ë
_tf_keras_layer±{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 6]}}
ô	

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
__call__
+&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 32]}}
è
4	variables
5regularization_losses
6trainable_variables
7	keras_api
__call__
+&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ö

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
__call__
+&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3136}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3136]}}
í

>kernel
?bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
__call__
+&call_and_return_all_conditional_losses"Æ
_tf_keras_layer¬{"class_name": "Dense", "name": "z_mu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_mu", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
û

Dkernel
Ebias
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
__call__
+&call_and_return_all_conditional_losses"Ô
_tf_keras_layerº{"class_name": "Dense", "name": "z_log_sigma", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_log_sigma", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
»
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
__call__
+&call_and_return_all_conditional_losses"ª
_tf_keras_layer{"class_name": "Sampling", "name": "sampling_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "sampling_1", "trainable": true, "dtype": "float32"}}
 "
trackable_list_wrapper
f
(0
)1
.2
/3
84
95
>6
?7
D8
E9"
trackable_list_wrapper
f
(0
)1
.2
/3
84
95
>6
?7
D8
E9"
trackable_list_wrapper
°
Nlayer_metrics

Olayers
Player_regularization_losses
Qmetrics
regularization_losses
	variables
Rnon_trainable_variables
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ô

Skernel
Tbias
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
__call__
+&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 3136, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
ú
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
__call__
+&call_and_return_all_conditional_losses"é
_tf_keras_layerÏ{"class_name": "Reshape", "name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 64]}}}
§


]kernel
^bias
_	variables
`regularization_losses
atrainable_variables
b	keras_api
__call__
+&call_and_return_all_conditional_losses"	
_tf_keras_layeræ{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 64]}}
©


ckernel
dbias
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
 __call__
+¡&call_and_return_all_conditional_losses"	
_tf_keras_layerè{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 64]}}
«


ikernel
jbias
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"	
_tf_keras_layerê{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_5", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 32]}}
 "
trackable_list_wrapper
X
S0
T1
]2
^3
c4
d5
i6
j7"
trackable_list_wrapper
X
S0
T1
]2
^3
c4
d5
i6
j7"
trackable_list_wrapper
°
olayer_metrics

players
qlayer_regularization_losses
rmetrics
regularization_losses
	variables
snon_trainable_variables
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ó

tkernel
ubias
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses"Ì
_tf_keras_layer²{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
ô

zkernel
{bias
|	variables
}regularization_losses
~trainable_variables
	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
÷
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 3, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
 "
trackable_list_wrapper
L
t0
u1
z2
{3
4
5"
trackable_list_wrapper
L
t0
u1
z2
{3
4
5"
trackable_list_wrapper
µ
layer_metrics
layers
 layer_regularization_losses
metrics
regularization_losses
 	variables
non_trainable_variables
!trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
1:/ 2encoder/conv2d_2/kernel
#:! 2encoder/conv2d_2/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
µ
*	variables
layer_metrics
 layer_regularization_losses
metrics
+regularization_losses
layers
non_trainable_variables
,trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
1:/ @2encoder/conv2d_3/kernel
#:!@2encoder/conv2d_3/bias
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
µ
0	variables
layer_metrics
 layer_regularization_losses
metrics
1regularization_losses
layers
non_trainable_variables
2trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
4	variables
layer_metrics
 layer_regularization_losses
metrics
5regularization_losses
layers
non_trainable_variables
6trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'	À2encoder/dense_5/kernel
": 2encoder/dense_5/bias
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
µ
:	variables
layer_metrics
 layer_regularization_losses
metrics
;regularization_losses
layers
non_trainable_variables
<trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
%:#2encoder/z_mu/kernel
:2encoder/z_mu/bias
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
µ
@	variables
layer_metrics
  layer_regularization_losses
¡metrics
Aregularization_losses
¢layers
£non_trainable_variables
Btrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
,:*2encoder/z_log_sigma/kernel
&:$2encoder/z_log_sigma/bias
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
µ
F	variables
¤layer_metrics
 ¥layer_regularization_losses
¦metrics
Gregularization_losses
§layers
¨non_trainable_variables
Htrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
J	variables
©layer_metrics
 ªlayer_regularization_losses
«metrics
Kregularization_losses
¬layers
­non_trainable_variables
Ltrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
Q
0
	1

2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
):'	
À2decoder/dense_6/kernel
#:!À2decoder/dense_6/bias
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
µ
U	variables
®layer_metrics
 ¯layer_regularization_losses
°metrics
Vregularization_losses
±layers
²non_trainable_variables
Wtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Y	variables
³layer_metrics
 ´layer_regularization_losses
µmetrics
Zregularization_losses
¶layers
·non_trainable_variables
[trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
;:9@@2!decoder/conv2d_transpose_3/kernel
-:+@2decoder/conv2d_transpose_3/bias
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
µ
_	variables
¸layer_metrics
 ¹layer_regularization_losses
ºmetrics
`regularization_losses
»layers
¼non_trainable_variables
atrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
;:9 @2!decoder/conv2d_transpose_4/kernel
-:+ 2decoder/conv2d_transpose_4/bias
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
µ
e	variables
½layer_metrics
 ¾layer_regularization_losses
¿metrics
fregularization_losses
Àlayers
Ánon_trainable_variables
gtrainable_variables
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
;:9 2!decoder/conv2d_transpose_5/kernel
-:+2decoder/conv2d_transpose_5/bias
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
µ
k	variables
Âlayer_metrics
 Ãlayer_regularization_losses
Ämetrics
lregularization_losses
Ålayers
Ænon_trainable_variables
mtrainable_variables
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
*:(
d2regressor/dense_7/kernel
$:"d2regressor/dense_7/bias
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
µ
v	variables
Çlayer_metrics
 Èlayer_regularization_losses
Émetrics
wregularization_losses
Êlayers
Ënon_trainable_variables
xtrainable_variables
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
*:(d2regressor/dense_8/kernel
$:"2regressor/dense_8/bias
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
µ
|	variables
Ìlayer_metrics
 Ílayer_regularization_losses
Îmetrics
}regularization_losses
Ïlayers
Ðnon_trainable_variables
~trainable_variables
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
*:(2regressor/dense_9/kernel
$:"2regressor/dense_9/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
	variables
Ñlayer_metrics
 Òlayer_regularization_losses
Ómetrics
regularization_losses
Ôlayers
Õnon_trainable_variables
trainable_variables
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
5
0
1
2"
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
6:4 2Adam/encoder/conv2d_2/kernel/m
(:& 2Adam/encoder/conv2d_2/bias/m
6:4 @2Adam/encoder/conv2d_3/kernel/m
(:&@2Adam/encoder/conv2d_3/bias/m
.:,	À2Adam/encoder/dense_5/kernel/m
':%2Adam/encoder/dense_5/bias/m
*:(2Adam/encoder/z_mu/kernel/m
$:"2Adam/encoder/z_mu/bias/m
1:/2!Adam/encoder/z_log_sigma/kernel/m
+:)2Adam/encoder/z_log_sigma/bias/m
.:,	
À2Adam/decoder/dense_6/kernel/m
(:&À2Adam/decoder/dense_6/bias/m
@:>@@2(Adam/decoder/conv2d_transpose_3/kernel/m
2:0@2&Adam/decoder/conv2d_transpose_3/bias/m
@:> @2(Adam/decoder/conv2d_transpose_4/kernel/m
2:0 2&Adam/decoder/conv2d_transpose_4/bias/m
@:> 2(Adam/decoder/conv2d_transpose_5/kernel/m
2:02&Adam/decoder/conv2d_transpose_5/bias/m
/:-
d2Adam/regressor/dense_7/kernel/m
):'d2Adam/regressor/dense_7/bias/m
/:-d2Adam/regressor/dense_8/kernel/m
):'2Adam/regressor/dense_8/bias/m
/:-2Adam/regressor/dense_9/kernel/m
):'2Adam/regressor/dense_9/bias/m
6:4 2Adam/encoder/conv2d_2/kernel/v
(:& 2Adam/encoder/conv2d_2/bias/v
6:4 @2Adam/encoder/conv2d_3/kernel/v
(:&@2Adam/encoder/conv2d_3/bias/v
.:,	À2Adam/encoder/dense_5/kernel/v
':%2Adam/encoder/dense_5/bias/v
*:(2Adam/encoder/z_mu/kernel/v
$:"2Adam/encoder/z_mu/bias/v
1:/2!Adam/encoder/z_log_sigma/kernel/v
+:)2Adam/encoder/z_log_sigma/bias/v
.:,	
À2Adam/decoder/dense_6/kernel/v
(:&À2Adam/decoder/dense_6/bias/v
@:>@@2(Adam/decoder/conv2d_transpose_3/kernel/v
2:0@2&Adam/decoder/conv2d_transpose_3/bias/v
@:> @2(Adam/decoder/conv2d_transpose_4/kernel/v
2:0 2&Adam/decoder/conv2d_transpose_4/bias/v
@:> 2(Adam/decoder/conv2d_transpose_5/kernel/v
2:02&Adam/decoder/conv2d_transpose_5/bias/v
/:-
d2Adam/regressor/dense_7/kernel/v
):'d2Adam/regressor/dense_7/bias/v
/:-d2Adam/regressor/dense_8/kernel/v
):'2Adam/regressor/dense_8/bias/v
/:-2Adam/regressor/dense_9/kernel/v
):'2Adam/regressor/dense_9/bias/v
þ2û
(__inference_encoder_layer_call_fn_119180Î
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
2
C__inference_encoder_layer_call_and_return_conditional_losses_119150Î
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
ö2ó
(__inference_decoder_layer_call_fn_119406Æ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

2
C__inference_decoder_layer_call_and_return_conditional_losses_119384Æ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

ø2õ
*__inference_regressor_layer_call_fn_119510Æ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

2
E__inference_regressor_layer_call_and_return_conditional_losses_119492Æ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

Ó2Ð
)__inference_conv2d_2_layer_call_fn_119530¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_2_layer_call_and_return_conditional_losses_119521¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_3_layer_call_fn_119550¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_3_layer_call_and_return_conditional_losses_119541¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_flatten_1_layer_call_fn_119561¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_flatten_1_layer_call_and_return_conditional_losses_119556¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_5_layer_call_fn_119581¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_5_layer_call_and_return_conditional_losses_119572¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_z_mu_layer_call_fn_119600¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_z_mu_layer_call_and_return_conditional_losses_119591¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_z_log_sigma_layer_call_fn_119619¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_z_log_sigma_layer_call_and_return_conditional_losses_119610¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_sampling_1_layer_call_fn_119651¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_sampling_1_layer_call_and_return_conditional_losses_119645¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_6_layer_call_fn_119671¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_6_layer_call_and_return_conditional_losses_119662¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_reshape_1_layer_call_fn_119690¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_reshape_1_layer_call_and_return_conditional_losses_119685¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
3__inference_conv2d_transpose_3_layer_call_fn_119225×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
­2ª
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_119215×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
2
3__inference_conv2d_transpose_4_layer_call_fn_119270×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
­2ª
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_119260×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
2
3__inference_conv2d_transpose_5_layer_call_fn_119315×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
­2ª
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_119305×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
Ò2Ï
(__inference_dense_7_layer_call_fn_119710¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_7_layer_call_and_return_conditional_losses_119701¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_8_layer_call_fn_119730¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_8_layer_call_and_return_conditional_losses_119721¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_9_layer_call_fn_119750¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_9_layer_call_and_return_conditional_losses_119741¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ´
D__inference_conv2d_2_layer_call_and_return_conditional_losses_119521l()7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_conv2d_2_layer_call_fn_119530_()7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ ´
D__inference_conv2d_3_layer_call_and_return_conditional_losses_119541l./7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_conv2d_3_layer_call_fn_119550_./7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@ã
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_119215]^I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 »
3__inference_conv2d_transpose_3_layer_call_fn_119225]^I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ã
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_119260cdI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 »
3__inference_conv2d_transpose_4_layer_call_fn_119270cdI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ã
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_119305ijI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 »
3__inference_conv2d_transpose_5_layer_call_fn_119315ijI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÄ
C__inference_decoder_layer_call_and_return_conditional_losses_119384}ST]^cdij0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
(__inference_decoder_layer_call_fn_119406pST]^cdij0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_5_layer_call_and_return_conditional_losses_119572]890¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_dense_5_layer_call_fn_119581P890¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_6_layer_call_and_return_conditional_losses_119662]ST/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "&¢#

0ÿÿÿÿÿÿÿÿÿÀ
 |
(__inference_dense_6_layer_call_fn_119671PST/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿÀ£
C__inference_dense_7_layer_call_and_return_conditional_losses_119701\tu/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 {
(__inference_dense_7_layer_call_fn_119710Otu/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿd£
C__inference_dense_8_layer_call_and_return_conditional_losses_119721\z{/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_8_layer_call_fn_119730Oz{/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿ¥
C__inference_dense_9_layer_call_and_return_conditional_losses_119741^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_dense_9_layer_call_fn_119750Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿú
C__inference_encoder_layer_call_and_return_conditional_losses_119150²
()./89>?DE8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
ª "j¢g
`¢]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 Ï
(__inference_encoder_layer_call_fn_119180¢
()./89>?DE8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
ª "Z¢W

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿª
E__inference_flatten_1_layer_call_and_return_conditional_losses_119556a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÀ
 
*__inference_flatten_1_layer_call_fn_119561T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿÀ¬
E__inference_regressor_layer_call_and_return_conditional_losses_119492ctuz{0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_regressor_layer_call_fn_119510Vtuz{0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿª
E__inference_reshape_1_layer_call_and_return_conditional_losses_119685a0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_reshape_1_layer_call_fn_119690T0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª " ÿÿÿÿÿÿÿÿÿ@Î
F__inference_sampling_1_layer_call_and_return_conditional_losses_119645Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¥
+__inference_sampling_1_layer_call_fn_119651vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_z_log_sigma_layer_call_and_return_conditional_losses_119610\DE/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_z_log_sigma_layer_call_fn_119619ODE/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ 
@__inference_z_mu_layer_call_and_return_conditional_losses_119591\>?/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 x
%__inference_z_mu_layer_call_fn_119600O>?/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
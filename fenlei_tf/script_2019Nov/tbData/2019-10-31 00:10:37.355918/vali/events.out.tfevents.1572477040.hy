       �K"	   �n�Abrain.Event:2=�IZ�,     @��	w�n�A"��
Y
xPlaceholder*
shape: *
dtype0*-
_output_shapes
:�����������
S
yPlaceholder*
shape: *
dtype0*'
_output_shapes
:���������
R
	keep_probPlaceholder*
shape:*
dtype0*
_output_shapes
:
l
random_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
W
random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
Y
random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
seed2*
dtype0*
seed���)*&
_output_shapes
:*
T0
�
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0*&
_output_shapes
:
l
random_normalAddrandom_normal/mulrandom_normal/mean*
T0*&
_output_shapes
:
�
wc1Variable*
shape:*
dtype0*&
_output_shapes
:*
	container *
shared_name 
�

wc1/AssignAssignwc1random_normal*
T0*
use_locking(*
validate_shape(*&
_output_shapes
:*
_class

loc:@wc1
b
wc1/readIdentitywc1*
T0*&
_output_shapes
:*
_class

loc:@wc1
n
random_normal_1/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
Y
random_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
seed2*
dtype0*
seed���)*&
_output_shapes
:*
T0
�
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
T0*&
_output_shapes
:
r
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
T0*&
_output_shapes
:
�
wc2Variable*
shape:*
dtype0*&
_output_shapes
:*
	container *
shared_name 
�

wc2/AssignAssignwc2random_normal_1*
T0*
use_locking(*
validate_shape(*&
_output_shapes
:*
_class

loc:@wc2
b
wc2/readIdentitywc2*
T0*&
_output_shapes
:*
_class

loc:@wc2
n
random_normal_2/shapeConst*
dtype0*
_output_shapes
:*%
valueB"             
Y
random_normal_2/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_2/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*
seed2*
dtype0*
seed���)*&
_output_shapes
: *
T0
�
random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
T0*&
_output_shapes
: 
r
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
T0*&
_output_shapes
: 
�
wc3Variable*
shape: *
dtype0*&
_output_shapes
: *
	container *
shared_name 
�

wc3/AssignAssignwc3random_normal_2*
T0*
use_locking(*
validate_shape(*&
_output_shapes
: *
_class

loc:@wc3
b
wc3/readIdentitywc3*
T0*&
_output_shapes
: *
_class

loc:@wc3
f
random_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"�  h  
Y
random_normal_3/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*
seed2*
dtype0*
seed���)* 
_output_shapes
:
�	�*
T0
�
random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev*
T0* 
_output_shapes
:
�	�
l
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean*
T0* 
_output_shapes
:
�	�
y
wd1Variable*
shape:
�	�*
dtype0* 
_output_shapes
:
�	�*
	container *
shared_name 
�

wd1/AssignAssignwd1random_normal_3*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
�	�*
_class

loc:@wd1
\
wd1/readIdentitywd1*
T0* 
_output_shapes
:
�	�*
_class

loc:@wd1
f
random_normal_4/shapeConst*
dtype0*
_output_shapes
:*
valueB"h     
Y
random_normal_4/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_4/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
$random_normal_4/RandomStandardNormalRandomStandardNormalrandom_normal_4/shape*
seed2*
dtype0*
seed���)*
_output_shapes
:	�*
T0
�
random_normal_4/mulMul$random_normal_4/RandomStandardNormalrandom_normal_4/stddev*
T0*
_output_shapes
:	�
k
random_normal_4Addrandom_normal_4/mulrandom_normal_4/mean*
T0*
_output_shapes
:	�
y
w_outVariable*
shape:	�*
dtype0*
_output_shapes
:	�*
	container *
shared_name 
�
w_out/AssignAssignw_outrandom_normal_4*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	�*
_class

loc:@w_out
a

w_out/readIdentityw_out*
T0*
_output_shapes
:	�*
_class

loc:@w_out
_
random_normal_5/shapeConst*
dtype0*
_output_shapes
:*
valueB:
Y
random_normal_5/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_5/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_5/RandomStandardNormalRandomStandardNormalrandom_normal_5/shape*
seed2 *
dtype0*

seed *
_output_shapes
:*
T0
}
random_normal_5/mulMul$random_normal_5/RandomStandardNormalrandom_normal_5/stddev*
T0*
_output_shapes
:
f
random_normal_5Addrandom_normal_5/mulrandom_normal_5/mean*
T0*
_output_shapes
:
m
bc1Variable*
shape:*
dtype0*
_output_shapes
:*
	container *
shared_name 
�

bc1/AssignAssignbc1random_normal_5*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*
_class

loc:@bc1
V
bc1/readIdentitybc1*
T0*
_output_shapes
:*
_class

loc:@bc1
_
random_normal_6/shapeConst*
dtype0*
_output_shapes
:*
valueB:
Y
random_normal_6/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_6/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_6/RandomStandardNormalRandomStandardNormalrandom_normal_6/shape*
seed2 *
dtype0*

seed *
_output_shapes
:*
T0
}
random_normal_6/mulMul$random_normal_6/RandomStandardNormalrandom_normal_6/stddev*
T0*
_output_shapes
:
f
random_normal_6Addrandom_normal_6/mulrandom_normal_6/mean*
T0*
_output_shapes
:
m
bc2Variable*
shape:*
dtype0*
_output_shapes
:*
	container *
shared_name 
�

bc2/AssignAssignbc2random_normal_6*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*
_class

loc:@bc2
V
bc2/readIdentitybc2*
T0*
_output_shapes
:*
_class

loc:@bc2
_
random_normal_7/shapeConst*
dtype0*
_output_shapes
:*
valueB: 
Y
random_normal_7/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_7/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_7/RandomStandardNormalRandomStandardNormalrandom_normal_7/shape*
seed2 *
dtype0*

seed *
_output_shapes
: *
T0
}
random_normal_7/mulMul$random_normal_7/RandomStandardNormalrandom_normal_7/stddev*
T0*
_output_shapes
: 
f
random_normal_7Addrandom_normal_7/mulrandom_normal_7/mean*
T0*
_output_shapes
: 
m
bc3Variable*
shape: *
dtype0*
_output_shapes
: *
	container *
shared_name 
�

bc3/AssignAssignbc3random_normal_7*
T0*
use_locking(*
validate_shape(*
_output_shapes
: *
_class

loc:@bc3
V
bc3/readIdentitybc3*
T0*
_output_shapes
: *
_class

loc:@bc3
`
random_normal_8/shapeConst*
dtype0*
_output_shapes
:*
valueB:�
Y
random_normal_8/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_8/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_8/RandomStandardNormalRandomStandardNormalrandom_normal_8/shape*
seed2 *
dtype0*

seed *
_output_shapes	
:�*
T0
~
random_normal_8/mulMul$random_normal_8/RandomStandardNormalrandom_normal_8/stddev*
T0*
_output_shapes	
:�
g
random_normal_8Addrandom_normal_8/mulrandom_normal_8/mean*
T0*
_output_shapes	
:�
o
bd1Variable*
shape:�*
dtype0*
_output_shapes	
:�*
	container *
shared_name 
�

bd1/AssignAssignbd1random_normal_8*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:�*
_class

loc:@bd1
W
bd1/readIdentitybd1*
T0*
_output_shapes	
:�*
_class

loc:@bd1
_
random_normal_9/shapeConst*
dtype0*
_output_shapes
:*
valueB:
Y
random_normal_9/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_9/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_9/RandomStandardNormalRandomStandardNormalrandom_normal_9/shape*
seed2 *
dtype0*

seed *
_output_shapes
:*
T0
}
random_normal_9/mulMul$random_normal_9/RandomStandardNormalrandom_normal_9/stddev*
T0*
_output_shapes
:
f
random_normal_9Addrandom_normal_9/mulrandom_normal_9/mean*
T0*
_output_shapes
:
o
b_outVariable*
shape:*
dtype0*
_output_shapes
:*
	container *
shared_name 
�
b_out/AssignAssignb_outrandom_normal_9*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*
_class

loc:@b_out
\

b_out/readIdentityb_out*
T0*
_output_shapes
:*
_class

loc:@b_out
f
Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"�����   �      
n
ReshapeReshapexReshape/shape*
T0*
Tshape0*1
_output_shapes
:�����������
�
Conv2DConv2DReshapewc1/read*
data_formatNHWC*/
_output_shapes
:���������..*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
u
BiasAddBiasAddConv2Dbc1/read*
T0*
data_formatNHWC*/
_output_shapes
:���������..
O
ReluReluBiasAdd*
T0*/
_output_shapes
:���������..
�
MaxPoolMaxPoolRelu*
strides
*
data_formatNHWC*/
_output_shapes
:���������*
ksize
*
T0*
paddingSAME
]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
strided_sliceStridedSlice	keep_probstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
: *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
T
dropout/ShapeShapeMaxPool*
T0*
out_type0*
_output_shapes
:
_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
_
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
seed2 *
dtype0*

seed */
_output_shapes
:���������*
T0
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*/
_output_shapes
:���������
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*/
_output_shapes
:���������
s
dropout/addAddstrided_slicedropout/random_uniform*
T0*/
_output_shapes
:���������
]
dropout/FloorFloordropout/add*
T0*/
_output_shapes
:���������
d
dropout/DivDivMaxPoolstrided_slice*
T0*/
_output_shapes
:���������
h
dropout/mulMuldropout/Divdropout/Floor*
T0*/
_output_shapes
:���������
�
Conv2D_1Conv2Ddropout/mulwc2/read*
data_formatNHWC*/
_output_shapes
:���������*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
y
	BiasAdd_1BiasAddConv2D_1bc2/read*
T0*
data_formatNHWC*/
_output_shapes
:���������
S
Relu_1Relu	BiasAdd_1*
T0*/
_output_shapes
:���������
�
	MaxPool_1MaxPoolRelu_1*
strides
*
data_formatNHWC*/
_output_shapes
:���������*
ksize
*
T0*
paddingSAME
_
strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
strided_slice_1StridedSlice	keep_probstrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
: *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
X
dropout_1/ShapeShape	MaxPool_1*
T0*
out_type0*
_output_shapes
:
a
dropout_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
a
dropout_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape*
seed2 *
dtype0*

seed */
_output_shapes
:���������*
T0
�
dropout_1/random_uniform/subSubdropout_1/random_uniform/maxdropout_1/random_uniform/min*
T0*
_output_shapes
: 
�
dropout_1/random_uniform/mulMul&dropout_1/random_uniform/RandomUniformdropout_1/random_uniform/sub*
T0*/
_output_shapes
:���������
�
dropout_1/random_uniformAdddropout_1/random_uniform/muldropout_1/random_uniform/min*
T0*/
_output_shapes
:���������
y
dropout_1/addAddstrided_slice_1dropout_1/random_uniform*
T0*/
_output_shapes
:���������
a
dropout_1/FloorFloordropout_1/add*
T0*/
_output_shapes
:���������
j
dropout_1/DivDiv	MaxPool_1strided_slice_1*
T0*/
_output_shapes
:���������
n
dropout_1/mulMuldropout_1/Divdropout_1/Floor*
T0*/
_output_shapes
:���������
�
Conv2D_2Conv2Ddropout_1/mulwc3/read*
data_formatNHWC*/
_output_shapes
:��������� *
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
y
	BiasAdd_2BiasAddConv2D_2bc3/read*
T0*
data_formatNHWC*/
_output_shapes
:��������� 
S
Relu_2Relu	BiasAdd_2*
T0*/
_output_shapes
:��������� 
�
	MaxPool_2MaxPoolRelu_2*
strides
*
data_formatNHWC*/
_output_shapes
:��������� *
ksize
*
T0*
paddingSAME
_
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
strided_slice_2StridedSlice	keep_probstrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
: *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
X
dropout_2/ShapeShape	MaxPool_2*
T0*
out_type0*
_output_shapes
:
a
dropout_2/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
a
dropout_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape*
seed2 *
dtype0*

seed */
_output_shapes
:��������� *
T0
�
dropout_2/random_uniform/subSubdropout_2/random_uniform/maxdropout_2/random_uniform/min*
T0*
_output_shapes
: 
�
dropout_2/random_uniform/mulMul&dropout_2/random_uniform/RandomUniformdropout_2/random_uniform/sub*
T0*/
_output_shapes
:��������� 
�
dropout_2/random_uniformAdddropout_2/random_uniform/muldropout_2/random_uniform/min*
T0*/
_output_shapes
:��������� 
y
dropout_2/addAddstrided_slice_2dropout_2/random_uniform*
T0*/
_output_shapes
:��������� 
a
dropout_2/FloorFloordropout_2/add*
T0*/
_output_shapes
:��������� 
j
dropout_2/DivDiv	MaxPool_2strided_slice_2*
T0*/
_output_shapes
:��������� 
n
dropout_2/mulMuldropout_2/Divdropout_2/Floor*
T0*/
_output_shapes
:��������� 
`
Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"�����  
u
	Reshape_1Reshapedropout_2/mulReshape_1/shape*
T0*
Tshape0*(
_output_shapes
:����������	
~
MatMulMatMul	Reshape_1wd1/read*
T0*
transpose_b( *
transpose_a( *(
_output_shapes
:����������
O
AddAddMatMulbd1/read*
T0*(
_output_shapes
:����������
F
Relu_3ReluAdd*
T0*(
_output_shapes
:����������
_
strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
strided_slice_3StridedSlice	keep_probstrided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
: *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
U
dropout_3/ShapeShapeRelu_3*
T0*
out_type0*
_output_shapes
:
a
dropout_3/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
a
dropout_3/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape*
seed2 *
dtype0*

seed *(
_output_shapes
:����������*
T0
�
dropout_3/random_uniform/subSubdropout_3/random_uniform/maxdropout_3/random_uniform/min*
T0*
_output_shapes
: 
�
dropout_3/random_uniform/mulMul&dropout_3/random_uniform/RandomUniformdropout_3/random_uniform/sub*
T0*(
_output_shapes
:����������
�
dropout_3/random_uniformAdddropout_3/random_uniform/muldropout_3/random_uniform/min*
T0*(
_output_shapes
:����������
r
dropout_3/addAddstrided_slice_3dropout_3/random_uniform*
T0*(
_output_shapes
:����������
Z
dropout_3/FloorFloordropout_3/add*
T0*(
_output_shapes
:����������
`
dropout_3/DivDivRelu_3strided_slice_3*
T0*(
_output_shapes
:����������
g
dropout_3/mulMuldropout_3/Divdropout_3/Floor*
T0*(
_output_shapes
:����������
�
MatMul_1MatMuldropout_3/mul
w_out/read*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:���������
T
Add_1AddMatMul_1
b_out/read*
T0*'
_output_shapes
:���������
K
pred/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
L
predAddAdd_1pred/y*
T0*'
_output_shapes
:���������
F
RankConst*
dtype0*
_output_shapes
: *
value	B :
I
ShapeShapepred*
T0*
out_type0*
_output_shapes
:
H
Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
K
Shape_1Shapepred*
T0*
out_type0*
_output_shapes
:
G
Sub/yConst*
dtype0*
_output_shapes
: *
value	B :
:
SubSubRank_1Sub/y*
T0*
_output_shapes
: 
R
Slice/beginPackSub*
N*
T0*
_output_shapes
:*

axis 
T

Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
b
SliceSliceShape_1Slice/begin
Slice/size*
T0*
Index0*
_output_shapes
:
S
concat/concat_dimConst*
dtype0*
_output_shapes
: *
value	B : 
b
concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
i
concatConcatconcat/concat_dimconcat/values_0Slice*
N*
_output_shapes
:*
T0
k
	Reshape_2Reshapepredconcat*
T0*
Tshape0*0
_output_shapes
:������������������
H
Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
H
Shape_2Shapey*
T0*
out_type0*
_output_shapes
:
I
Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
>
Sub_1SubRank_2Sub_1/y*
T0*
_output_shapes
: 
V
Slice_1/beginPackSub_1*
N*
T0*
_output_shapes
:*

axis 
V
Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
T0*
Index0*
_output_shapes
:
U
concat_1/concat_dimConst*
dtype0*
_output_shapes
: *
value	B : 
d
concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
q
concat_1Concatconcat_1/concat_dimconcat_1/values_0Slice_1*
N*
_output_shapes
:*
T0
j
	Reshape_3Reshapeyconcat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*
T0*?
_output_shapes-
+:���������:������������������
I
Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
<
Sub_2SubRankSub_2/y*
T0*
_output_shapes
: 
W
Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB: 
U
Slice_2/sizePackSub_2*
N*
T0*
_output_shapes
:*

axis 
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
T0*
Index0*#
_output_shapes
:���������
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
T0*
Tshape0*#
_output_shapes
:���������
O
ConstConst*
dtype0*
_output_shapes
:*
valueB: 
\
costMean	Reshape_4Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
T
gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/cost_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
gradients/cost_grad/ReshapeReshapegradients/Fill!gradients/cost_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
b
gradients/cost_grad/ShapeShape	Reshape_4*
T0*
out_type0*
_output_shapes
:
�
gradients/cost_grad/TileTilegradients/cost_grad/Reshapegradients/cost_grad/Shape*
T0*#
_output_shapes
:���������*

Tmultiples0
d
gradients/cost_grad/Shape_1Shape	Reshape_4*
T0*
out_type0*
_output_shapes
:
^
gradients/cost_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
c
gradients/cost_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
gradients/cost_grad/ProdProdgradients/cost_grad/Shape_1gradients/cost_grad/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
e
gradients/cost_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
gradients/cost_grad/Prod_1Prodgradients/cost_grad/Shape_2gradients/cost_grad/Const_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
_
gradients/cost_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
gradients/cost_grad/MaximumMaximumgradients/cost_grad/Prod_1gradients/cost_grad/Maximum/y*
T0*
_output_shapes
: 
{
gradients/cost_grad/floordivDivgradients/cost_grad/Prodgradients/cost_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/cost_grad/CastCastgradients/cost_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
gradients/cost_grad/truedivDivgradients/cost_grad/Tilegradients/cost_grad/Cast*
T0*#
_output_shapes
:���������
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
�
 gradients/Reshape_4_grad/ReshapeReshapegradients/cost_grad/truedivgradients/Reshape_4_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
b
gradients/Reshape_2_grad/ShapeShapepred*
T0*
out_type0*
_output_shapes
:
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
^
gradients/pred_grad/ShapeShapeAdd_1*
T0*
out_type0*
_output_shapes
:
^
gradients/pred_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
)gradients/pred_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pred_grad/Shapegradients/pred_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/pred_grad/SumSum gradients/Reshape_2_grad/Reshape)gradients/pred_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
gradients/pred_grad/ReshapeReshapegradients/pred_grad/Sumgradients/pred_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/pred_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape+gradients/pred_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
gradients/pred_grad/Reshape_1Reshapegradients/pred_grad/Sum_1gradients/pred_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
j
$gradients/pred_grad/tuple/group_depsNoOp^gradients/pred_grad/Reshape^gradients/pred_grad/Reshape_1
�
,gradients/pred_grad/tuple/control_dependencyIdentitygradients/pred_grad/Reshape%^gradients/pred_grad/tuple/group_deps*
T0*'
_output_shapes
:���������*.
_class$
" loc:@gradients/pred_grad/Reshape
�
.gradients/pred_grad/tuple/control_dependency_1Identitygradients/pred_grad/Reshape_1%^gradients/pred_grad/tuple/group_deps*
T0*
_output_shapes
: *0
_class&
$"loc:@gradients/pred_grad/Reshape_1
b
gradients/Add_1_grad/ShapeShapeMatMul_1*
T0*
out_type0*
_output_shapes
:
f
gradients/Add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Add_1_grad/SumSum,gradients/pred_grad/tuple/control_dependency*gradients/Add_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/Add_1_grad/Sum_1Sum,gradients/pred_grad/tuple/control_dependency,gradients/Add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
�
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*
T0*'
_output_shapes
:���������*/
_class%
#!loc:@gradients/Add_1_grad/Reshape
�
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*
T0*
_output_shapes
:*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/Add_1_grad/tuple/control_dependency
w_out/read*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:����������
�
 gradients/MatMul_1_grad/MatMul_1MatMuldropout_3/mul-gradients/Add_1_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes
:	�
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*
_output_shapes
:	�*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1
o
"gradients/dropout_3/mul_grad/ShapeShapedropout_3/Div*
T0*
out_type0*
_output_shapes
:
s
$gradients/dropout_3/mul_grad/Shape_1Shapedropout_3/Floor*
T0*
out_type0*
_output_shapes
:
�
2gradients/dropout_3/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_3/mul_grad/Shape$gradients/dropout_3/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
 gradients/dropout_3/mul_grad/mulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout_3/Floor*
T0*(
_output_shapes
:����������
�
 gradients/dropout_3/mul_grad/SumSum gradients/dropout_3/mul_grad/mul2gradients/dropout_3/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
$gradients/dropout_3/mul_grad/ReshapeReshape gradients/dropout_3/mul_grad/Sum"gradients/dropout_3/mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
"gradients/dropout_3/mul_grad/mul_1Muldropout_3/Div0gradients/MatMul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
"gradients/dropout_3/mul_grad/Sum_1Sum"gradients/dropout_3/mul_grad/mul_14gradients/dropout_3/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
&gradients/dropout_3/mul_grad/Reshape_1Reshape"gradients/dropout_3/mul_grad/Sum_1$gradients/dropout_3/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
-gradients/dropout_3/mul_grad/tuple/group_depsNoOp%^gradients/dropout_3/mul_grad/Reshape'^gradients/dropout_3/mul_grad/Reshape_1
�
5gradients/dropout_3/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_3/mul_grad/Reshape.^gradients/dropout_3/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*7
_class-
+)loc:@gradients/dropout_3/mul_grad/Reshape
�
7gradients/dropout_3/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_3/mul_grad/Reshape_1.^gradients/dropout_3/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*9
_class/
-+loc:@gradients/dropout_3/mul_grad/Reshape_1
h
"gradients/dropout_3/Div_grad/ShapeShapeRelu_3*
T0*
out_type0*
_output_shapes
:
g
$gradients/dropout_3/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
2gradients/dropout_3/Div_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_3/Div_grad/Shape$gradients/dropout_3/Div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/dropout_3/Div_grad/truedivDiv5gradients/dropout_3/mul_grad/tuple/control_dependencystrided_slice_3*
T0*(
_output_shapes
:����������
�
 gradients/dropout_3/Div_grad/SumSum$gradients/dropout_3/Div_grad/truediv2gradients/dropout_3/Div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
$gradients/dropout_3/Div_grad/ReshapeReshape gradients/dropout_3/Div_grad/Sum"gradients/dropout_3/Div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
b
 gradients/dropout_3/Div_grad/NegNegRelu_3*
T0*(
_output_shapes
:����������
_
#gradients/dropout_3/Div_grad/SquareSquarestrided_slice_3*
T0*
_output_shapes
: 
�
&gradients/dropout_3/Div_grad/truediv_1Div gradients/dropout_3/Div_grad/Neg#gradients/dropout_3/Div_grad/Square*
T0*(
_output_shapes
:����������
�
 gradients/dropout_3/Div_grad/mulMul5gradients/dropout_3/mul_grad/tuple/control_dependency&gradients/dropout_3/Div_grad/truediv_1*
T0*(
_output_shapes
:����������
�
"gradients/dropout_3/Div_grad/Sum_1Sum gradients/dropout_3/Div_grad/mul4gradients/dropout_3/Div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
&gradients/dropout_3/Div_grad/Reshape_1Reshape"gradients/dropout_3/Div_grad/Sum_1$gradients/dropout_3/Div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
-gradients/dropout_3/Div_grad/tuple/group_depsNoOp%^gradients/dropout_3/Div_grad/Reshape'^gradients/dropout_3/Div_grad/Reshape_1
�
5gradients/dropout_3/Div_grad/tuple/control_dependencyIdentity$gradients/dropout_3/Div_grad/Reshape.^gradients/dropout_3/Div_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*7
_class-
+)loc:@gradients/dropout_3/Div_grad/Reshape
�
7gradients/dropout_3/Div_grad/tuple/control_dependency_1Identity&gradients/dropout_3/Div_grad/Reshape_1.^gradients/dropout_3/Div_grad/tuple/group_deps*
T0*
_output_shapes
: *9
_class/
-+loc:@gradients/dropout_3/Div_grad/Reshape_1
�
gradients/Relu_3_grad/ReluGradReluGrad5gradients/dropout_3/Div_grad/tuple/control_dependencyRelu_3*
T0*(
_output_shapes
:����������
^
gradients/Add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
e
gradients/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Add_grad/SumSumgradients/Relu_3_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
gradients/Add_grad/Sum_1Sumgradients/Relu_3_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
�
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*-
_class#
!loc:@gradients/Add_grad/Reshape
�
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*
T0*
_output_shapes	
:�*/
_class%
#!loc:@gradients/Add_grad/Reshape_1
�
gradients/MatMul_grad/MatMulMatMul+gradients/Add_grad/tuple/control_dependencywd1/read*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:����������	
�
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1+gradients/Add_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(* 
_output_shapes
:
�	�
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������	*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
�	�*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
k
gradients/Reshape_1_grad/ShapeShapedropout_2/mul*
T0*
out_type0*
_output_shapes
:
�
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*
Tshape0*/
_output_shapes
:��������� 
o
"gradients/dropout_2/mul_grad/ShapeShapedropout_2/Div*
T0*
out_type0*
_output_shapes
:
s
$gradients/dropout_2/mul_grad/Shape_1Shapedropout_2/Floor*
T0*
out_type0*
_output_shapes
:
�
2gradients/dropout_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_2/mul_grad/Shape$gradients/dropout_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
 gradients/dropout_2/mul_grad/mulMul gradients/Reshape_1_grad/Reshapedropout_2/Floor*
T0*/
_output_shapes
:��������� 
�
 gradients/dropout_2/mul_grad/SumSum gradients/dropout_2/mul_grad/mul2gradients/dropout_2/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
$gradients/dropout_2/mul_grad/ReshapeReshape gradients/dropout_2/mul_grad/Sum"gradients/dropout_2/mul_grad/Shape*
T0*
Tshape0*/
_output_shapes
:��������� 
�
"gradients/dropout_2/mul_grad/mul_1Muldropout_2/Div gradients/Reshape_1_grad/Reshape*
T0*/
_output_shapes
:��������� 
�
"gradients/dropout_2/mul_grad/Sum_1Sum"gradients/dropout_2/mul_grad/mul_14gradients/dropout_2/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
&gradients/dropout_2/mul_grad/Reshape_1Reshape"gradients/dropout_2/mul_grad/Sum_1$gradients/dropout_2/mul_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:��������� 
�
-gradients/dropout_2/mul_grad/tuple/group_depsNoOp%^gradients/dropout_2/mul_grad/Reshape'^gradients/dropout_2/mul_grad/Reshape_1
�
5gradients/dropout_2/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_2/mul_grad/Reshape.^gradients/dropout_2/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:��������� *7
_class-
+)loc:@gradients/dropout_2/mul_grad/Reshape
�
7gradients/dropout_2/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_2/mul_grad/Reshape_1.^gradients/dropout_2/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:��������� *9
_class/
-+loc:@gradients/dropout_2/mul_grad/Reshape_1
k
"gradients/dropout_2/Div_grad/ShapeShape	MaxPool_2*
T0*
out_type0*
_output_shapes
:
g
$gradients/dropout_2/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
2gradients/dropout_2/Div_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_2/Div_grad/Shape$gradients/dropout_2/Div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/dropout_2/Div_grad/truedivDiv5gradients/dropout_2/mul_grad/tuple/control_dependencystrided_slice_2*
T0*/
_output_shapes
:��������� 
�
 gradients/dropout_2/Div_grad/SumSum$gradients/dropout_2/Div_grad/truediv2gradients/dropout_2/Div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
$gradients/dropout_2/Div_grad/ReshapeReshape gradients/dropout_2/Div_grad/Sum"gradients/dropout_2/Div_grad/Shape*
T0*
Tshape0*/
_output_shapes
:��������� 
l
 gradients/dropout_2/Div_grad/NegNeg	MaxPool_2*
T0*/
_output_shapes
:��������� 
_
#gradients/dropout_2/Div_grad/SquareSquarestrided_slice_2*
T0*
_output_shapes
: 
�
&gradients/dropout_2/Div_grad/truediv_1Div gradients/dropout_2/Div_grad/Neg#gradients/dropout_2/Div_grad/Square*
T0*/
_output_shapes
:��������� 
�
 gradients/dropout_2/Div_grad/mulMul5gradients/dropout_2/mul_grad/tuple/control_dependency&gradients/dropout_2/Div_grad/truediv_1*
T0*/
_output_shapes
:��������� 
�
"gradients/dropout_2/Div_grad/Sum_1Sum gradients/dropout_2/Div_grad/mul4gradients/dropout_2/Div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
&gradients/dropout_2/Div_grad/Reshape_1Reshape"gradients/dropout_2/Div_grad/Sum_1$gradients/dropout_2/Div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
-gradients/dropout_2/Div_grad/tuple/group_depsNoOp%^gradients/dropout_2/Div_grad/Reshape'^gradients/dropout_2/Div_grad/Reshape_1
�
5gradients/dropout_2/Div_grad/tuple/control_dependencyIdentity$gradients/dropout_2/Div_grad/Reshape.^gradients/dropout_2/Div_grad/tuple/group_deps*
T0*/
_output_shapes
:��������� *7
_class-
+)loc:@gradients/dropout_2/Div_grad/Reshape
�
7gradients/dropout_2/Div_grad/tuple/control_dependency_1Identity&gradients/dropout_2/Div_grad/Reshape_1.^gradients/dropout_2/Div_grad/tuple/group_deps*
T0*
_output_shapes
: *9
_class/
-+loc:@gradients/dropout_2/Div_grad/Reshape_1
�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_25gradients/dropout_2/Div_grad/tuple/control_dependency*
strides
*
data_formatNHWC*/
_output_shapes
:��������� *
ksize
*
T0*
paddingSAME
�
gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*
T0*/
_output_shapes
:��������� 
�
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
�
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*/
_output_shapes
:��������� *1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad
�
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*
_output_shapes
: *7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad
j
gradients/Conv2D_2_grad/ShapeShapedropout_1/mul*
T0*
out_type0*
_output_shapes
:
�
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/Shapewc3/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
x
gradients/Conv2D_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"             
�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterdropout_1/mulgradients/Conv2D_2_grad/Shape_11gradients/BiasAdd_2_grad/tuple/control_dependency*
data_formatNHWC*&
_output_shapes
: *
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*&
_output_shapes
: *?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter
o
"gradients/dropout_1/mul_grad/ShapeShapedropout_1/Div*
T0*
out_type0*
_output_shapes
:
s
$gradients/dropout_1/mul_grad/Shape_1Shapedropout_1/Floor*
T0*
out_type0*
_output_shapes
:
�
2gradients/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_1/mul_grad/Shape$gradients/dropout_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
 gradients/dropout_1/mul_grad/mulMul0gradients/Conv2D_2_grad/tuple/control_dependencydropout_1/Floor*
T0*/
_output_shapes
:���������
�
 gradients/dropout_1/mul_grad/SumSum gradients/dropout_1/mul_grad/mul2gradients/dropout_1/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
$gradients/dropout_1/mul_grad/ReshapeReshape gradients/dropout_1/mul_grad/Sum"gradients/dropout_1/mul_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
�
"gradients/dropout_1/mul_grad/mul_1Muldropout_1/Div0gradients/Conv2D_2_grad/tuple/control_dependency*
T0*/
_output_shapes
:���������
�
"gradients/dropout_1/mul_grad/Sum_1Sum"gradients/dropout_1/mul_grad/mul_14gradients/dropout_1/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
&gradients/dropout_1/mul_grad/Reshape_1Reshape"gradients/dropout_1/mul_grad/Sum_1$gradients/dropout_1/mul_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:���������
�
-gradients/dropout_1/mul_grad/tuple/group_depsNoOp%^gradients/dropout_1/mul_grad/Reshape'^gradients/dropout_1/mul_grad/Reshape_1
�
5gradients/dropout_1/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_1/mul_grad/Reshape.^gradients/dropout_1/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*7
_class-
+)loc:@gradients/dropout_1/mul_grad/Reshape
�
7gradients/dropout_1/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_1/mul_grad/Reshape_1.^gradients/dropout_1/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*9
_class/
-+loc:@gradients/dropout_1/mul_grad/Reshape_1
k
"gradients/dropout_1/Div_grad/ShapeShape	MaxPool_1*
T0*
out_type0*
_output_shapes
:
g
$gradients/dropout_1/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
2gradients/dropout_1/Div_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_1/Div_grad/Shape$gradients/dropout_1/Div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/dropout_1/Div_grad/truedivDiv5gradients/dropout_1/mul_grad/tuple/control_dependencystrided_slice_1*
T0*/
_output_shapes
:���������
�
 gradients/dropout_1/Div_grad/SumSum$gradients/dropout_1/Div_grad/truediv2gradients/dropout_1/Div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
$gradients/dropout_1/Div_grad/ReshapeReshape gradients/dropout_1/Div_grad/Sum"gradients/dropout_1/Div_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
l
 gradients/dropout_1/Div_grad/NegNeg	MaxPool_1*
T0*/
_output_shapes
:���������
_
#gradients/dropout_1/Div_grad/SquareSquarestrided_slice_1*
T0*
_output_shapes
: 
�
&gradients/dropout_1/Div_grad/truediv_1Div gradients/dropout_1/Div_grad/Neg#gradients/dropout_1/Div_grad/Square*
T0*/
_output_shapes
:���������
�
 gradients/dropout_1/Div_grad/mulMul5gradients/dropout_1/mul_grad/tuple/control_dependency&gradients/dropout_1/Div_grad/truediv_1*
T0*/
_output_shapes
:���������
�
"gradients/dropout_1/Div_grad/Sum_1Sum gradients/dropout_1/Div_grad/mul4gradients/dropout_1/Div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
&gradients/dropout_1/Div_grad/Reshape_1Reshape"gradients/dropout_1/Div_grad/Sum_1$gradients/dropout_1/Div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
-gradients/dropout_1/Div_grad/tuple/group_depsNoOp%^gradients/dropout_1/Div_grad/Reshape'^gradients/dropout_1/Div_grad/Reshape_1
�
5gradients/dropout_1/Div_grad/tuple/control_dependencyIdentity$gradients/dropout_1/Div_grad/Reshape.^gradients/dropout_1/Div_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*7
_class-
+)loc:@gradients/dropout_1/Div_grad/Reshape
�
7gradients/dropout_1/Div_grad/tuple/control_dependency_1Identity&gradients/dropout_1/Div_grad/Reshape_1.^gradients/dropout_1/Div_grad/tuple/group_deps*
T0*
_output_shapes
: *9
_class/
-+loc:@gradients/dropout_1/Div_grad/Reshape_1
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_15gradients/dropout_1/Div_grad/tuple/control_dependency*
strides
*
data_formatNHWC*/
_output_shapes
:���������*
ksize
*
T0*
paddingSAME
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*/
_output_shapes
:���������
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*
_output_shapes
:*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad
h
gradients/Conv2D_1_grad/ShapeShapedropout/mul*
T0*
out_type0*
_output_shapes
:
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/Shapewc2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
x
gradients/Conv2D_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterdropout/mulgradients/Conv2D_1_grad/Shape_11gradients/BiasAdd_1_grad/tuple/control_dependency*
data_formatNHWC*&
_output_shapes
:*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*&
_output_shapes
:*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
k
 gradients/dropout/mul_grad/ShapeShapedropout/Div*
T0*
out_type0*
_output_shapes
:
o
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
T0*
out_type0*
_output_shapes
:
�
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/dropout/mul_grad/mulMul0gradients/Conv2D_1_grad/tuple/control_dependencydropout/Floor*
T0*/
_output_shapes
:���������
�
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
�
 gradients/dropout/mul_grad/mul_1Muldropout/Div0gradients/Conv2D_1_grad/tuple/control_dependency*
T0*/
_output_shapes
:���������
�
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:���������

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
�
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape
�
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1
g
 gradients/dropout/Div_grad/ShapeShapeMaxPool*
T0*
out_type0*
_output_shapes
:
e
"gradients/dropout/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
0gradients/dropout/Div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/Div_grad/Shape"gradients/dropout/Div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
"gradients/dropout/Div_grad/truedivDiv3gradients/dropout/mul_grad/tuple/control_dependencystrided_slice*
T0*/
_output_shapes
:���������
�
gradients/dropout/Div_grad/SumSum"gradients/dropout/Div_grad/truediv0gradients/dropout/Div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
"gradients/dropout/Div_grad/ReshapeReshapegradients/dropout/Div_grad/Sum gradients/dropout/Div_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
h
gradients/dropout/Div_grad/NegNegMaxPool*
T0*/
_output_shapes
:���������
[
!gradients/dropout/Div_grad/SquareSquarestrided_slice*
T0*
_output_shapes
: 
�
$gradients/dropout/Div_grad/truediv_1Divgradients/dropout/Div_grad/Neg!gradients/dropout/Div_grad/Square*
T0*/
_output_shapes
:���������
�
gradients/dropout/Div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/Div_grad/truediv_1*
T0*/
_output_shapes
:���������
�
 gradients/dropout/Div_grad/Sum_1Sumgradients/dropout/Div_grad/mul2gradients/dropout/Div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
$gradients/dropout/Div_grad/Reshape_1Reshape gradients/dropout/Div_grad/Sum_1"gradients/dropout/Div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

+gradients/dropout/Div_grad/tuple/group_depsNoOp#^gradients/dropout/Div_grad/Reshape%^gradients/dropout/Div_grad/Reshape_1
�
3gradients/dropout/Div_grad/tuple/control_dependencyIdentity"gradients/dropout/Div_grad/Reshape,^gradients/dropout/Div_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*5
_class+
)'loc:@gradients/dropout/Div_grad/Reshape
�
5gradients/dropout/Div_grad/tuple/control_dependency_1Identity$gradients/dropout/Div_grad/Reshape_1,^gradients/dropout/Div_grad/tuple/group_deps*
T0*
_output_shapes
: *7
_class-
+)loc:@gradients/dropout/Div_grad/Reshape_1
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool3gradients/dropout/Div_grad/tuple/control_dependency*
strides
*
data_formatNHWC*/
_output_shapes
:���������..*
ksize
*
T0*
paddingSAME
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*/
_output_shapes
:���������..
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*/
_output_shapes
:���������..*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad
b
gradients/Conv2D_grad/ShapeShapeReshape*
T0*
out_type0*
_output_shapes
:
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/Shapewc1/read/gradients/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
v
gradients/Conv2D_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/Shape_1/gradients/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*&
_output_shapes
:*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*1
_output_shapes
:�����������*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*&
_output_shapes
:*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter
b
GradientDescent/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *�Q8=
�
/GradientDescent/update_wc1/ApplyGradientDescentApplyGradientDescentwc1GradientDescent/learning_rate0gradients/Conv2D_grad/tuple/control_dependency_1*
T0*
use_locking( *&
_output_shapes
:*
_class

loc:@wc1
�
/GradientDescent/update_wc2/ApplyGradientDescentApplyGradientDescentwc2GradientDescent/learning_rate2gradients/Conv2D_1_grad/tuple/control_dependency_1*
T0*
use_locking( *&
_output_shapes
:*
_class

loc:@wc2
�
/GradientDescent/update_wc3/ApplyGradientDescentApplyGradientDescentwc3GradientDescent/learning_rate2gradients/Conv2D_2_grad/tuple/control_dependency_1*
T0*
use_locking( *&
_output_shapes
: *
_class

loc:@wc3
�
/GradientDescent/update_wd1/ApplyGradientDescentApplyGradientDescentwd1GradientDescent/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( * 
_output_shapes
:
�	�*
_class

loc:@wd1
�
1GradientDescent/update_w_out/ApplyGradientDescentApplyGradientDescentw_outGradientDescent/learning_rate2gradients/MatMul_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes
:	�*
_class

loc:@w_out
�
/GradientDescent/update_bc1/ApplyGradientDescentApplyGradientDescentbc1GradientDescent/learning_rate1gradients/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes
:*
_class

loc:@bc1
�
/GradientDescent/update_bc2/ApplyGradientDescentApplyGradientDescentbc2GradientDescent/learning_rate3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes
:*
_class

loc:@bc2
�
/GradientDescent/update_bc3/ApplyGradientDescentApplyGradientDescentbc3GradientDescent/learning_rate3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes
: *
_class

loc:@bc3
�
/GradientDescent/update_bd1/ApplyGradientDescentApplyGradientDescentbd1GradientDescent/learning_rate-gradients/Add_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes	
:�*
_class

loc:@bd1
�
1GradientDescent/update_b_out/ApplyGradientDescentApplyGradientDescentb_outGradientDescent/learning_rate/gradients/Add_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes
:*
_class

loc:@b_out
�
GradientDescentNoOp0^GradientDescent/update_wc1/ApplyGradientDescent0^GradientDescent/update_wc2/ApplyGradientDescent0^GradientDescent/update_wc3/ApplyGradientDescent0^GradientDescent/update_wd1/ApplyGradientDescent2^GradientDescent/update_w_out/ApplyGradientDescent0^GradientDescent/update_bc1/ApplyGradientDescent0^GradientDescent/update_bc2/ApplyGradientDescent0^GradientDescent/update_bc3/ApplyGradientDescent0^GradientDescent/update_bd1/ApplyGradientDescent2^GradientDescent/update_b_out/ApplyGradientDescent
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
b
ArgMaxArgMaxpredArgMax/dimension*
T0*#
_output_shapes
:���������*

Tidx0
T
ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
c
ArgMax_1ArgMaxyArgMax_1/dimension*
T0*#
_output_shapes
:���������*

Tidx0
T
ArgMax_2/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
f
ArgMax_2ArgMaxpredArgMax_2/dimension*
T0*#
_output_shapes
:���������*

Tidx0
T
ArgMax_3/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
c
ArgMax_3ArgMaxyArgMax_3/dimension*
T0*#
_output_shapes
:���������*

Tidx0
P
EqualEqualArgMax_2ArgMax_3*
T0	*#
_output_shapes
:���������
R
Cast_1CastEqual*

SrcT0
*#
_output_shapes
:���������*

DstT0
Q
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
_
accuracyMeanCast_1Const_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
[
ScalarSummary/tagsConst*
dtype0*
_output_shapes
: *
valueB BAccuracy
]
ScalarSummaryScalarSummaryScalarSummary/tagsaccuracy*
T0*
_output_shapes
: 
Y
ScalarSummary_1/tagsConst*
dtype0*
_output_shapes
: *
valueB
 BLoss
]
ScalarSummary_1ScalarSummaryScalarSummary_1/tagscost*
T0*
_output_shapes
: 
h
Reshape_5/shapeConst*
dtype0*
_output_shapes
:*%
valueB"�����   �      
r
	Reshape_5ReshapexReshape_5/shape*
T0*
Tshape0*1
_output_shapes
:�����������
Y
ImageSummary/tagConst*
dtype0*
_output_shapes
: *
valueB BOriginal
�
ImageSummaryImageSummaryImageSummary/tag	Reshape_5*
T0*
_output_shapes
: *
	bad_colorB:�  �*

max_images
h
Reshape_6/shapeConst*
dtype0*
_output_shapes
:*%
valueB"�����   �      
r
	Reshape_6ReshapexReshape_6/shape*
T0*
Tshape0*1
_output_shapes
:�����������
�
Conv2D_3Conv2D	Reshape_6wc1/read*
data_formatNHWC*/
_output_shapes
:���������..*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
y
	BiasAdd_3BiasAddConv2D_3bc1/read*
T0*
data_formatNHWC*/
_output_shapes
:���������..
S
Relu_4Relu	BiasAdd_3*
T0*/
_output_shapes
:���������..
L
conv1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
W
conv1AddRelu_4conv1/y*
T0*/
_output_shapes
:���������..
h
Reshape_7/shapeConst*
dtype0*
_output_shapes
:*%
valueB"����.   .      
t
	Reshape_7Reshapeconv1Reshape_7/shape*
T0*
Tshape0*/
_output_shapes
:���������..
Y
ImageSummary_1/tagConst*
dtype0*
_output_shapes
: *
valueB B1.Conv
�
ImageSummary_1ImageSummaryImageSummary_1/tag	Reshape_7*
T0*
_output_shapes
: *
	bad_colorB:�  �*

max_images
�
Conv2D_4Conv2Dconv1wc2/read*
data_formatNHWC*/
_output_shapes
:���������..*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
y
	BiasAdd_4BiasAddConv2D_4bc2/read*
T0*
data_formatNHWC*/
_output_shapes
:���������..
S
Relu_5Relu	BiasAdd_4*
T0*/
_output_shapes
:���������..
L
conv2/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
W
conv2AddRelu_5conv2/y*
T0*/
_output_shapes
:���������..
h
Reshape_8/shapeConst*
dtype0*
_output_shapes
:*%
valueB"����.   .      
t
	Reshape_8Reshapeconv2Reshape_8/shape*
T0*
Tshape0*/
_output_shapes
:���������..
Y
ImageSummary_2/tagConst*
dtype0*
_output_shapes
: *
valueB B2.Conv
�
ImageSummary_2ImageSummaryImageSummary_2/tag	Reshape_8*
T0*
_output_shapes
: *
	bad_colorB:�  �*

max_images
�
Conv2D_5Conv2Dconv2wc3/read*
data_formatNHWC*/
_output_shapes
:���������.. *
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
y
	BiasAdd_5BiasAddConv2D_5bc3/read*
T0*
data_formatNHWC*/
_output_shapes
:���������.. 
S
Relu_6Relu	BiasAdd_5*
T0*/
_output_shapes
:���������.. 
L
conv3/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
W
conv3AddRelu_6conv3/y*
T0*/
_output_shapes
:���������.. 
h
Reshape_9/shapeConst*
dtype0*
_output_shapes
:*%
valueB"����.   .      
t
	Reshape_9Reshapeconv3Reshape_9/shape*
T0*
Tshape0*/
_output_shapes
:���������..
Y
ImageSummary_3/tagConst*
dtype0*
_output_shapes
: *
valueB B3.Conv
�
ImageSummary_3ImageSummaryImageSummary_3/tag	Reshape_9*
T0*
_output_shapes
: *
	bad_colorB:�  �*

max_images 
e
HistogramSummary/tagConst*
dtype0*
_output_shapes
: *!
valueB BHistogram 1.Conv
e
HistogramSummaryHistogramSummaryHistogramSummary/tagwc1/read*
T0*
_output_shapes
: 
e
HistogramSummary_1/tagConst*
dtype0*
_output_shapes
: *
valueB BHistogram pred
e
HistogramSummary_1HistogramSummaryHistogramSummary_1/tagpred*
T0*
_output_shapes
: 
�
MergeSummary/MergeSummaryMergeSummaryScalarSummaryScalarSummary_1ImageSummaryImageSummary_1ImageSummary_2ImageSummary_3HistogramSummaryHistogramSummary_1*
N*
_output_shapes
: 
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
�
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:
*I
value@B>
Bb_outBbc1Bbc2Bbc3Bbd1Bw_outBwc1Bwc2Bwc3Bwd1
w
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:
*'
valueB
B B B B B B B B B B 
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesb_outbc1bc2bc3bd1w_outwc1wc2wc3wd1*
dtypes
2

}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_output_shapes
: *
_class
loc:@save/Const
i
save/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBb_out
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/AssignAssignb_outsave/RestoreV2*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*
_class

loc:@b_out
i
save/RestoreV2_1/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBbc1
j
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_1Assignbc1save/RestoreV2_1*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*
_class

loc:@bc1
i
save/RestoreV2_2/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBbc2
j
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_2Assignbc2save/RestoreV2_2*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*
_class

loc:@bc2
i
save/RestoreV2_3/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBbc3
j
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_3Assignbc3save/RestoreV2_3*
T0*
use_locking(*
validate_shape(*
_output_shapes
: *
_class

loc:@bc3
i
save/RestoreV2_4/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBbd1
j
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_4Assignbd1save/RestoreV2_4*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:�*
_class

loc:@bd1
k
save/RestoreV2_5/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBw_out
j
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_5Assignw_outsave/RestoreV2_5*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	�*
_class

loc:@w_out
i
save/RestoreV2_6/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBwc1
j
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_6Assignwc1save/RestoreV2_6*
T0*
use_locking(*
validate_shape(*&
_output_shapes
:*
_class

loc:@wc1
i
save/RestoreV2_7/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBwc2
j
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_7Assignwc2save/RestoreV2_7*
T0*
use_locking(*
validate_shape(*&
_output_shapes
:*
_class

loc:@wc2
i
save/RestoreV2_8/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBwc3
j
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_8Assignwc3save/RestoreV2_8*
T0*
use_locking(*
validate_shape(*&
_output_shapes
: *
_class

loc:@wc3
i
save/RestoreV2_9/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBwd1
j
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_9Assignwd1save/RestoreV2_9*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
�	�*
_class

loc:@wd1
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^wc1/Assign^wc2/Assign^wc3/Assign^wd1/Assign^w_out/Assign^bc1/Assign^bc2/Assign^bc3/Assign^bd1/Assign^b_out/Assign"�-5E     U��	�EG�n�AJ��
�)�)
9
Add
x"T
y"T
z"T"
Ttype:
2	
�
ApplyGradientDescent
var"T�

alpha"T

delta"T
out"T�"
Ttype:
2	"
use_lockingbool( 
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
N
Concat

concat_dim
values"T*N
output"T"
Nint(0"	
Ttype
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
9
Div
x"T
y"T
z"T"
Ttype:
2	
A
Equal
x"T
y"T
z
"
Ttype:
2	
�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
+
Floor
x"T
y"T"
Ttype:
2
S
HistogramSummary
tag
values"T
summary"
Ttype0:
2		
.
Identity

input"T
output"T"	
Ttype
�
ImageSummary
tag
tensor"T
summary"

max_imagesint(0"
Ttype0:
2"'
	bad_colortensorB:�  �
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
�
MaxPool

input"T
output"T"
Ttype0:
2"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
<
Mul
x"T
y"T
z"T"
Ttype:
2	�
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
l
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
i
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
i
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
0
Square
x"T
y"T"
Ttype:
	2	
�
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
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
Variable
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*0.12.020.12.0-rc1-7-ga13284f-dirty��
Y
xPlaceholder*
shape: *
dtype0*-
_output_shapes
:�����������
S
yPlaceholder*
shape: *
dtype0*'
_output_shapes
:���������
R
	keep_probPlaceholder*
shape:*
dtype0*
_output_shapes
:
l
random_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
W
random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
Y
random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
seed2*
dtype0*
seed���)*&
_output_shapes
:*
T0
�
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0*&
_output_shapes
:
l
random_normalAddrandom_normal/mulrandom_normal/mean*
T0*&
_output_shapes
:
�
wc1Variable*
shape:*
dtype0*
	container *&
_output_shapes
:*
shared_name 
�

wc1/AssignAssignwc1random_normal*
T0*
use_locking(*
validate_shape(*&
_output_shapes
:*
_class

loc:@wc1
b
wc1/readIdentitywc1*
T0*&
_output_shapes
:*
_class

loc:@wc1
n
random_normal_1/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
Y
random_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
seed2*
dtype0*
seed���)*&
_output_shapes
:*
T0
�
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
T0*&
_output_shapes
:
r
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
T0*&
_output_shapes
:
�
wc2Variable*
shape:*
dtype0*
	container *&
_output_shapes
:*
shared_name 
�

wc2/AssignAssignwc2random_normal_1*
T0*
use_locking(*
validate_shape(*&
_output_shapes
:*
_class

loc:@wc2
b
wc2/readIdentitywc2*
T0*&
_output_shapes
:*
_class

loc:@wc2
n
random_normal_2/shapeConst*
dtype0*
_output_shapes
:*%
valueB"             
Y
random_normal_2/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_2/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*
seed2*
dtype0*
seed���)*&
_output_shapes
: *
T0
�
random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
T0*&
_output_shapes
: 
r
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
T0*&
_output_shapes
: 
�
wc3Variable*
shape: *
dtype0*
	container *&
_output_shapes
: *
shared_name 
�

wc3/AssignAssignwc3random_normal_2*
T0*
use_locking(*
validate_shape(*&
_output_shapes
: *
_class

loc:@wc3
b
wc3/readIdentitywc3*
T0*&
_output_shapes
: *
_class

loc:@wc3
f
random_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"�  h  
Y
random_normal_3/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*
seed2*
dtype0*
seed���)* 
_output_shapes
:
�	�*
T0
�
random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev*
T0* 
_output_shapes
:
�	�
l
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean*
T0* 
_output_shapes
:
�	�
y
wd1Variable*
shape:
�	�*
dtype0*
	container * 
_output_shapes
:
�	�*
shared_name 
�

wd1/AssignAssignwd1random_normal_3*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
�	�*
_class

loc:@wd1
\
wd1/readIdentitywd1*
T0* 
_output_shapes
:
�	�*
_class

loc:@wd1
f
random_normal_4/shapeConst*
dtype0*
_output_shapes
:*
valueB"h     
Y
random_normal_4/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_4/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
$random_normal_4/RandomStandardNormalRandomStandardNormalrandom_normal_4/shape*
seed2*
dtype0*
seed���)*
_output_shapes
:	�*
T0
�
random_normal_4/mulMul$random_normal_4/RandomStandardNormalrandom_normal_4/stddev*
T0*
_output_shapes
:	�
k
random_normal_4Addrandom_normal_4/mulrandom_normal_4/mean*
T0*
_output_shapes
:	�
y
w_outVariable*
shape:	�*
dtype0*
	container *
_output_shapes
:	�*
shared_name 
�
w_out/AssignAssignw_outrandom_normal_4*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	�*
_class

loc:@w_out
a

w_out/readIdentityw_out*
T0*
_output_shapes
:	�*
_class

loc:@w_out
_
random_normal_5/shapeConst*
dtype0*
_output_shapes
:*
valueB:
Y
random_normal_5/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_5/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_5/RandomStandardNormalRandomStandardNormalrandom_normal_5/shape*
seed2 *
dtype0*

seed *
_output_shapes
:*
T0
}
random_normal_5/mulMul$random_normal_5/RandomStandardNormalrandom_normal_5/stddev*
T0*
_output_shapes
:
f
random_normal_5Addrandom_normal_5/mulrandom_normal_5/mean*
T0*
_output_shapes
:
m
bc1Variable*
shape:*
dtype0*
	container *
_output_shapes
:*
shared_name 
�

bc1/AssignAssignbc1random_normal_5*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*
_class

loc:@bc1
V
bc1/readIdentitybc1*
T0*
_output_shapes
:*
_class

loc:@bc1
_
random_normal_6/shapeConst*
dtype0*
_output_shapes
:*
valueB:
Y
random_normal_6/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_6/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_6/RandomStandardNormalRandomStandardNormalrandom_normal_6/shape*
seed2 *
dtype0*

seed *
_output_shapes
:*
T0
}
random_normal_6/mulMul$random_normal_6/RandomStandardNormalrandom_normal_6/stddev*
T0*
_output_shapes
:
f
random_normal_6Addrandom_normal_6/mulrandom_normal_6/mean*
T0*
_output_shapes
:
m
bc2Variable*
shape:*
dtype0*
	container *
_output_shapes
:*
shared_name 
�

bc2/AssignAssignbc2random_normal_6*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*
_class

loc:@bc2
V
bc2/readIdentitybc2*
T0*
_output_shapes
:*
_class

loc:@bc2
_
random_normal_7/shapeConst*
dtype0*
_output_shapes
:*
valueB: 
Y
random_normal_7/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_7/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_7/RandomStandardNormalRandomStandardNormalrandom_normal_7/shape*
seed2 *
dtype0*

seed *
_output_shapes
: *
T0
}
random_normal_7/mulMul$random_normal_7/RandomStandardNormalrandom_normal_7/stddev*
T0*
_output_shapes
: 
f
random_normal_7Addrandom_normal_7/mulrandom_normal_7/mean*
T0*
_output_shapes
: 
m
bc3Variable*
shape: *
dtype0*
	container *
_output_shapes
: *
shared_name 
�

bc3/AssignAssignbc3random_normal_7*
T0*
use_locking(*
validate_shape(*
_output_shapes
: *
_class

loc:@bc3
V
bc3/readIdentitybc3*
T0*
_output_shapes
: *
_class

loc:@bc3
`
random_normal_8/shapeConst*
dtype0*
_output_shapes
:*
valueB:�
Y
random_normal_8/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_8/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_8/RandomStandardNormalRandomStandardNormalrandom_normal_8/shape*
seed2 *
dtype0*

seed *
_output_shapes	
:�*
T0
~
random_normal_8/mulMul$random_normal_8/RandomStandardNormalrandom_normal_8/stddev*
T0*
_output_shapes	
:�
g
random_normal_8Addrandom_normal_8/mulrandom_normal_8/mean*
T0*
_output_shapes	
:�
o
bd1Variable*
shape:�*
dtype0*
	container *
_output_shapes	
:�*
shared_name 
�

bd1/AssignAssignbd1random_normal_8*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:�*
_class

loc:@bd1
W
bd1/readIdentitybd1*
T0*
_output_shapes	
:�*
_class

loc:@bd1
_
random_normal_9/shapeConst*
dtype0*
_output_shapes
:*
valueB:
Y
random_normal_9/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_9/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_9/RandomStandardNormalRandomStandardNormalrandom_normal_9/shape*
seed2 *
dtype0*

seed *
_output_shapes
:*
T0
}
random_normal_9/mulMul$random_normal_9/RandomStandardNormalrandom_normal_9/stddev*
T0*
_output_shapes
:
f
random_normal_9Addrandom_normal_9/mulrandom_normal_9/mean*
T0*
_output_shapes
:
o
b_outVariable*
shape:*
dtype0*
	container *
_output_shapes
:*
shared_name 
�
b_out/AssignAssignb_outrandom_normal_9*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*
_class

loc:@b_out
\

b_out/readIdentityb_out*
T0*
_output_shapes
:*
_class

loc:@b_out
f
Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"�����   �      
n
ReshapeReshapexReshape/shape*
T0*
Tshape0*1
_output_shapes
:�����������
�
Conv2DConv2DReshapewc1/read*
data_formatNHWC*/
_output_shapes
:���������..*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
u
BiasAddBiasAddConv2Dbc1/read*
T0*
data_formatNHWC*/
_output_shapes
:���������..
O
ReluReluBiasAdd*
T0*/
_output_shapes
:���������..
�
MaxPoolMaxPoolRelu*
ksize
*
data_formatNHWC*/
_output_shapes
:���������*
strides
*
T0*
paddingSAME
]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
strided_sliceStridedSlice	keep_probstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
: *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
T
dropout/ShapeShapeMaxPool*
T0*
out_type0*
_output_shapes
:
_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
_
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
seed2 *
dtype0*

seed */
_output_shapes
:���������*
T0
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*/
_output_shapes
:���������
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*/
_output_shapes
:���������
s
dropout/addAddstrided_slicedropout/random_uniform*
T0*/
_output_shapes
:���������
]
dropout/FloorFloordropout/add*
T0*/
_output_shapes
:���������
d
dropout/DivDivMaxPoolstrided_slice*
T0*/
_output_shapes
:���������
h
dropout/mulMuldropout/Divdropout/Floor*
T0*/
_output_shapes
:���������
�
Conv2D_1Conv2Ddropout/mulwc2/read*
data_formatNHWC*/
_output_shapes
:���������*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
y
	BiasAdd_1BiasAddConv2D_1bc2/read*
T0*
data_formatNHWC*/
_output_shapes
:���������
S
Relu_1Relu	BiasAdd_1*
T0*/
_output_shapes
:���������
�
	MaxPool_1MaxPoolRelu_1*
ksize
*
data_formatNHWC*/
_output_shapes
:���������*
strides
*
T0*
paddingSAME
_
strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
strided_slice_1StridedSlice	keep_probstrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
: *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
X
dropout_1/ShapeShape	MaxPool_1*
T0*
out_type0*
_output_shapes
:
a
dropout_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
a
dropout_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape*
seed2 *
dtype0*

seed */
_output_shapes
:���������*
T0
�
dropout_1/random_uniform/subSubdropout_1/random_uniform/maxdropout_1/random_uniform/min*
T0*
_output_shapes
: 
�
dropout_1/random_uniform/mulMul&dropout_1/random_uniform/RandomUniformdropout_1/random_uniform/sub*
T0*/
_output_shapes
:���������
�
dropout_1/random_uniformAdddropout_1/random_uniform/muldropout_1/random_uniform/min*
T0*/
_output_shapes
:���������
y
dropout_1/addAddstrided_slice_1dropout_1/random_uniform*
T0*/
_output_shapes
:���������
a
dropout_1/FloorFloordropout_1/add*
T0*/
_output_shapes
:���������
j
dropout_1/DivDiv	MaxPool_1strided_slice_1*
T0*/
_output_shapes
:���������
n
dropout_1/mulMuldropout_1/Divdropout_1/Floor*
T0*/
_output_shapes
:���������
�
Conv2D_2Conv2Ddropout_1/mulwc3/read*
data_formatNHWC*/
_output_shapes
:��������� *
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
y
	BiasAdd_2BiasAddConv2D_2bc3/read*
T0*
data_formatNHWC*/
_output_shapes
:��������� 
S
Relu_2Relu	BiasAdd_2*
T0*/
_output_shapes
:��������� 
�
	MaxPool_2MaxPoolRelu_2*
ksize
*
data_formatNHWC*/
_output_shapes
:��������� *
strides
*
T0*
paddingSAME
_
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
strided_slice_2StridedSlice	keep_probstrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
: *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
X
dropout_2/ShapeShape	MaxPool_2*
T0*
out_type0*
_output_shapes
:
a
dropout_2/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
a
dropout_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape*
seed2 *
dtype0*

seed */
_output_shapes
:��������� *
T0
�
dropout_2/random_uniform/subSubdropout_2/random_uniform/maxdropout_2/random_uniform/min*
T0*
_output_shapes
: 
�
dropout_2/random_uniform/mulMul&dropout_2/random_uniform/RandomUniformdropout_2/random_uniform/sub*
T0*/
_output_shapes
:��������� 
�
dropout_2/random_uniformAdddropout_2/random_uniform/muldropout_2/random_uniform/min*
T0*/
_output_shapes
:��������� 
y
dropout_2/addAddstrided_slice_2dropout_2/random_uniform*
T0*/
_output_shapes
:��������� 
a
dropout_2/FloorFloordropout_2/add*
T0*/
_output_shapes
:��������� 
j
dropout_2/DivDiv	MaxPool_2strided_slice_2*
T0*/
_output_shapes
:��������� 
n
dropout_2/mulMuldropout_2/Divdropout_2/Floor*
T0*/
_output_shapes
:��������� 
`
Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"�����  
u
	Reshape_1Reshapedropout_2/mulReshape_1/shape*
T0*
Tshape0*(
_output_shapes
:����������	
~
MatMulMatMul	Reshape_1wd1/read*
T0*
transpose_b( *
transpose_a( *(
_output_shapes
:����������
O
AddAddMatMulbd1/read*
T0*(
_output_shapes
:����������
F
Relu_3ReluAdd*
T0*(
_output_shapes
:����������
_
strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
strided_slice_3StridedSlice	keep_probstrided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
: *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
U
dropout_3/ShapeShapeRelu_3*
T0*
out_type0*
_output_shapes
:
a
dropout_3/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
a
dropout_3/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape*
seed2 *
dtype0*

seed *(
_output_shapes
:����������*
T0
�
dropout_3/random_uniform/subSubdropout_3/random_uniform/maxdropout_3/random_uniform/min*
T0*
_output_shapes
: 
�
dropout_3/random_uniform/mulMul&dropout_3/random_uniform/RandomUniformdropout_3/random_uniform/sub*
T0*(
_output_shapes
:����������
�
dropout_3/random_uniformAdddropout_3/random_uniform/muldropout_3/random_uniform/min*
T0*(
_output_shapes
:����������
r
dropout_3/addAddstrided_slice_3dropout_3/random_uniform*
T0*(
_output_shapes
:����������
Z
dropout_3/FloorFloordropout_3/add*
T0*(
_output_shapes
:����������
`
dropout_3/DivDivRelu_3strided_slice_3*
T0*(
_output_shapes
:����������
g
dropout_3/mulMuldropout_3/Divdropout_3/Floor*
T0*(
_output_shapes
:����������
�
MatMul_1MatMuldropout_3/mul
w_out/read*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:���������
T
Add_1AddMatMul_1
b_out/read*
T0*'
_output_shapes
:���������
K
pred/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
L
predAddAdd_1pred/y*
T0*'
_output_shapes
:���������
F
RankConst*
dtype0*
_output_shapes
: *
value	B :
I
ShapeShapepred*
T0*
out_type0*
_output_shapes
:
H
Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
K
Shape_1Shapepred*
T0*
out_type0*
_output_shapes
:
G
Sub/yConst*
dtype0*
_output_shapes
: *
value	B :
:
SubSubRank_1Sub/y*
T0*
_output_shapes
: 
R
Slice/beginPackSub*
N*

axis *
_output_shapes
:*
T0
T

Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
b
SliceSliceShape_1Slice/begin
Slice/size*
T0*
Index0*
_output_shapes
:
S
concat/concat_dimConst*
dtype0*
_output_shapes
: *
value	B : 
b
concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
i
concatConcatconcat/concat_dimconcat/values_0Slice*
N*
_output_shapes
:*
T0
k
	Reshape_2Reshapepredconcat*
T0*
Tshape0*0
_output_shapes
:������������������
H
Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
H
Shape_2Shapey*
T0*
out_type0*
_output_shapes
:
I
Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
>
Sub_1SubRank_2Sub_1/y*
T0*
_output_shapes
: 
V
Slice_1/beginPackSub_1*
N*

axis *
_output_shapes
:*
T0
V
Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
T0*
Index0*
_output_shapes
:
U
concat_1/concat_dimConst*
dtype0*
_output_shapes
: *
value	B : 
d
concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
q
concat_1Concatconcat_1/concat_dimconcat_1/values_0Slice_1*
N*
_output_shapes
:*
T0
j
	Reshape_3Reshapeyconcat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*
T0*?
_output_shapes-
+:���������:������������������
I
Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
<
Sub_2SubRankSub_2/y*
T0*
_output_shapes
: 
W
Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB: 
U
Slice_2/sizePackSub_2*
N*

axis *
_output_shapes
:*
T0
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
T0*
Index0*#
_output_shapes
:���������
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
T0*
Tshape0*#
_output_shapes
:���������
O
ConstConst*
dtype0*
_output_shapes
:*
valueB: 
\
costMean	Reshape_4Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
T
gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/cost_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
gradients/cost_grad/ReshapeReshapegradients/Fill!gradients/cost_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
b
gradients/cost_grad/ShapeShape	Reshape_4*
T0*
out_type0*
_output_shapes
:
�
gradients/cost_grad/TileTilegradients/cost_grad/Reshapegradients/cost_grad/Shape*
T0*#
_output_shapes
:���������*

Tmultiples0
d
gradients/cost_grad/Shape_1Shape	Reshape_4*
T0*
out_type0*
_output_shapes
:
^
gradients/cost_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
c
gradients/cost_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
gradients/cost_grad/ProdProdgradients/cost_grad/Shape_1gradients/cost_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
e
gradients/cost_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
gradients/cost_grad/Prod_1Prodgradients/cost_grad/Shape_2gradients/cost_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
_
gradients/cost_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
gradients/cost_grad/MaximumMaximumgradients/cost_grad/Prod_1gradients/cost_grad/Maximum/y*
T0*
_output_shapes
: 
{
gradients/cost_grad/floordivDivgradients/cost_grad/Prodgradients/cost_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/cost_grad/CastCastgradients/cost_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
gradients/cost_grad/truedivDivgradients/cost_grad/Tilegradients/cost_grad/Cast*
T0*#
_output_shapes
:���������
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
�
 gradients/Reshape_4_grad/ReshapeReshapegradients/cost_grad/truedivgradients/Reshape_4_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
b
gradients/Reshape_2_grad/ShapeShapepred*
T0*
out_type0*
_output_shapes
:
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
^
gradients/pred_grad/ShapeShapeAdd_1*
T0*
out_type0*
_output_shapes
:
^
gradients/pred_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
)gradients/pred_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pred_grad/Shapegradients/pred_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/pred_grad/SumSum gradients/Reshape_2_grad/Reshape)gradients/pred_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/pred_grad/ReshapeReshapegradients/pred_grad/Sumgradients/pred_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/pred_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape+gradients/pred_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/pred_grad/Reshape_1Reshapegradients/pred_grad/Sum_1gradients/pred_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
j
$gradients/pred_grad/tuple/group_depsNoOp^gradients/pred_grad/Reshape^gradients/pred_grad/Reshape_1
�
,gradients/pred_grad/tuple/control_dependencyIdentitygradients/pred_grad/Reshape%^gradients/pred_grad/tuple/group_deps*
T0*'
_output_shapes
:���������*.
_class$
" loc:@gradients/pred_grad/Reshape
�
.gradients/pred_grad/tuple/control_dependency_1Identitygradients/pred_grad/Reshape_1%^gradients/pred_grad/tuple/group_deps*
T0*
_output_shapes
: *0
_class&
$"loc:@gradients/pred_grad/Reshape_1
b
gradients/Add_1_grad/ShapeShapeMatMul_1*
T0*
out_type0*
_output_shapes
:
f
gradients/Add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Add_1_grad/SumSum,gradients/pred_grad/tuple/control_dependency*gradients/Add_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/Add_1_grad/Sum_1Sum,gradients/pred_grad/tuple/control_dependency,gradients/Add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
�
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*
T0*'
_output_shapes
:���������*/
_class%
#!loc:@gradients/Add_1_grad/Reshape
�
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*
T0*
_output_shapes
:*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/Add_1_grad/tuple/control_dependency
w_out/read*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:����������
�
 gradients/MatMul_1_grad/MatMul_1MatMuldropout_3/mul-gradients/Add_1_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes
:	�
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*
_output_shapes
:	�*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1
o
"gradients/dropout_3/mul_grad/ShapeShapedropout_3/Div*
T0*
out_type0*
_output_shapes
:
s
$gradients/dropout_3/mul_grad/Shape_1Shapedropout_3/Floor*
T0*
out_type0*
_output_shapes
:
�
2gradients/dropout_3/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_3/mul_grad/Shape$gradients/dropout_3/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
 gradients/dropout_3/mul_grad/mulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout_3/Floor*
T0*(
_output_shapes
:����������
�
 gradients/dropout_3/mul_grad/SumSum gradients/dropout_3/mul_grad/mul2gradients/dropout_3/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
$gradients/dropout_3/mul_grad/ReshapeReshape gradients/dropout_3/mul_grad/Sum"gradients/dropout_3/mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
"gradients/dropout_3/mul_grad/mul_1Muldropout_3/Div0gradients/MatMul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
"gradients/dropout_3/mul_grad/Sum_1Sum"gradients/dropout_3/mul_grad/mul_14gradients/dropout_3/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
&gradients/dropout_3/mul_grad/Reshape_1Reshape"gradients/dropout_3/mul_grad/Sum_1$gradients/dropout_3/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
-gradients/dropout_3/mul_grad/tuple/group_depsNoOp%^gradients/dropout_3/mul_grad/Reshape'^gradients/dropout_3/mul_grad/Reshape_1
�
5gradients/dropout_3/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_3/mul_grad/Reshape.^gradients/dropout_3/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*7
_class-
+)loc:@gradients/dropout_3/mul_grad/Reshape
�
7gradients/dropout_3/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_3/mul_grad/Reshape_1.^gradients/dropout_3/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*9
_class/
-+loc:@gradients/dropout_3/mul_grad/Reshape_1
h
"gradients/dropout_3/Div_grad/ShapeShapeRelu_3*
T0*
out_type0*
_output_shapes
:
g
$gradients/dropout_3/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
2gradients/dropout_3/Div_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_3/Div_grad/Shape$gradients/dropout_3/Div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/dropout_3/Div_grad/truedivDiv5gradients/dropout_3/mul_grad/tuple/control_dependencystrided_slice_3*
T0*(
_output_shapes
:����������
�
 gradients/dropout_3/Div_grad/SumSum$gradients/dropout_3/Div_grad/truediv2gradients/dropout_3/Div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
$gradients/dropout_3/Div_grad/ReshapeReshape gradients/dropout_3/Div_grad/Sum"gradients/dropout_3/Div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
b
 gradients/dropout_3/Div_grad/NegNegRelu_3*
T0*(
_output_shapes
:����������
_
#gradients/dropout_3/Div_grad/SquareSquarestrided_slice_3*
T0*
_output_shapes
: 
�
&gradients/dropout_3/Div_grad/truediv_1Div gradients/dropout_3/Div_grad/Neg#gradients/dropout_3/Div_grad/Square*
T0*(
_output_shapes
:����������
�
 gradients/dropout_3/Div_grad/mulMul5gradients/dropout_3/mul_grad/tuple/control_dependency&gradients/dropout_3/Div_grad/truediv_1*
T0*(
_output_shapes
:����������
�
"gradients/dropout_3/Div_grad/Sum_1Sum gradients/dropout_3/Div_grad/mul4gradients/dropout_3/Div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
&gradients/dropout_3/Div_grad/Reshape_1Reshape"gradients/dropout_3/Div_grad/Sum_1$gradients/dropout_3/Div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
-gradients/dropout_3/Div_grad/tuple/group_depsNoOp%^gradients/dropout_3/Div_grad/Reshape'^gradients/dropout_3/Div_grad/Reshape_1
�
5gradients/dropout_3/Div_grad/tuple/control_dependencyIdentity$gradients/dropout_3/Div_grad/Reshape.^gradients/dropout_3/Div_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*7
_class-
+)loc:@gradients/dropout_3/Div_grad/Reshape
�
7gradients/dropout_3/Div_grad/tuple/control_dependency_1Identity&gradients/dropout_3/Div_grad/Reshape_1.^gradients/dropout_3/Div_grad/tuple/group_deps*
T0*
_output_shapes
: *9
_class/
-+loc:@gradients/dropout_3/Div_grad/Reshape_1
�
gradients/Relu_3_grad/ReluGradReluGrad5gradients/dropout_3/Div_grad/tuple/control_dependencyRelu_3*
T0*(
_output_shapes
:����������
^
gradients/Add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
e
gradients/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Add_grad/SumSumgradients/Relu_3_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
gradients/Add_grad/Sum_1Sumgradients/Relu_3_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
�
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*-
_class#
!loc:@gradients/Add_grad/Reshape
�
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*
T0*
_output_shapes	
:�*/
_class%
#!loc:@gradients/Add_grad/Reshape_1
�
gradients/MatMul_grad/MatMulMatMul+gradients/Add_grad/tuple/control_dependencywd1/read*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:����������	
�
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1+gradients/Add_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(* 
_output_shapes
:
�	�
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������	*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
�	�*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
k
gradients/Reshape_1_grad/ShapeShapedropout_2/mul*
T0*
out_type0*
_output_shapes
:
�
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*
Tshape0*/
_output_shapes
:��������� 
o
"gradients/dropout_2/mul_grad/ShapeShapedropout_2/Div*
T0*
out_type0*
_output_shapes
:
s
$gradients/dropout_2/mul_grad/Shape_1Shapedropout_2/Floor*
T0*
out_type0*
_output_shapes
:
�
2gradients/dropout_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_2/mul_grad/Shape$gradients/dropout_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
 gradients/dropout_2/mul_grad/mulMul gradients/Reshape_1_grad/Reshapedropout_2/Floor*
T0*/
_output_shapes
:��������� 
�
 gradients/dropout_2/mul_grad/SumSum gradients/dropout_2/mul_grad/mul2gradients/dropout_2/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
$gradients/dropout_2/mul_grad/ReshapeReshape gradients/dropout_2/mul_grad/Sum"gradients/dropout_2/mul_grad/Shape*
T0*
Tshape0*/
_output_shapes
:��������� 
�
"gradients/dropout_2/mul_grad/mul_1Muldropout_2/Div gradients/Reshape_1_grad/Reshape*
T0*/
_output_shapes
:��������� 
�
"gradients/dropout_2/mul_grad/Sum_1Sum"gradients/dropout_2/mul_grad/mul_14gradients/dropout_2/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
&gradients/dropout_2/mul_grad/Reshape_1Reshape"gradients/dropout_2/mul_grad/Sum_1$gradients/dropout_2/mul_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:��������� 
�
-gradients/dropout_2/mul_grad/tuple/group_depsNoOp%^gradients/dropout_2/mul_grad/Reshape'^gradients/dropout_2/mul_grad/Reshape_1
�
5gradients/dropout_2/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_2/mul_grad/Reshape.^gradients/dropout_2/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:��������� *7
_class-
+)loc:@gradients/dropout_2/mul_grad/Reshape
�
7gradients/dropout_2/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_2/mul_grad/Reshape_1.^gradients/dropout_2/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:��������� *9
_class/
-+loc:@gradients/dropout_2/mul_grad/Reshape_1
k
"gradients/dropout_2/Div_grad/ShapeShape	MaxPool_2*
T0*
out_type0*
_output_shapes
:
g
$gradients/dropout_2/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
2gradients/dropout_2/Div_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_2/Div_grad/Shape$gradients/dropout_2/Div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/dropout_2/Div_grad/truedivDiv5gradients/dropout_2/mul_grad/tuple/control_dependencystrided_slice_2*
T0*/
_output_shapes
:��������� 
�
 gradients/dropout_2/Div_grad/SumSum$gradients/dropout_2/Div_grad/truediv2gradients/dropout_2/Div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
$gradients/dropout_2/Div_grad/ReshapeReshape gradients/dropout_2/Div_grad/Sum"gradients/dropout_2/Div_grad/Shape*
T0*
Tshape0*/
_output_shapes
:��������� 
l
 gradients/dropout_2/Div_grad/NegNeg	MaxPool_2*
T0*/
_output_shapes
:��������� 
_
#gradients/dropout_2/Div_grad/SquareSquarestrided_slice_2*
T0*
_output_shapes
: 
�
&gradients/dropout_2/Div_grad/truediv_1Div gradients/dropout_2/Div_grad/Neg#gradients/dropout_2/Div_grad/Square*
T0*/
_output_shapes
:��������� 
�
 gradients/dropout_2/Div_grad/mulMul5gradients/dropout_2/mul_grad/tuple/control_dependency&gradients/dropout_2/Div_grad/truediv_1*
T0*/
_output_shapes
:��������� 
�
"gradients/dropout_2/Div_grad/Sum_1Sum gradients/dropout_2/Div_grad/mul4gradients/dropout_2/Div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
&gradients/dropout_2/Div_grad/Reshape_1Reshape"gradients/dropout_2/Div_grad/Sum_1$gradients/dropout_2/Div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
-gradients/dropout_2/Div_grad/tuple/group_depsNoOp%^gradients/dropout_2/Div_grad/Reshape'^gradients/dropout_2/Div_grad/Reshape_1
�
5gradients/dropout_2/Div_grad/tuple/control_dependencyIdentity$gradients/dropout_2/Div_grad/Reshape.^gradients/dropout_2/Div_grad/tuple/group_deps*
T0*/
_output_shapes
:��������� *7
_class-
+)loc:@gradients/dropout_2/Div_grad/Reshape
�
7gradients/dropout_2/Div_grad/tuple/control_dependency_1Identity&gradients/dropout_2/Div_grad/Reshape_1.^gradients/dropout_2/Div_grad/tuple/group_deps*
T0*
_output_shapes
: *9
_class/
-+loc:@gradients/dropout_2/Div_grad/Reshape_1
�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_25gradients/dropout_2/Div_grad/tuple/control_dependency*
ksize
*
data_formatNHWC*/
_output_shapes
:��������� *
strides
*
T0*
paddingSAME
�
gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*
T0*/
_output_shapes
:��������� 
�
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
�
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*/
_output_shapes
:��������� *1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad
�
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*
_output_shapes
: *7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad
j
gradients/Conv2D_2_grad/ShapeShapedropout_1/mul*
T0*
out_type0*
_output_shapes
:
�
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/Shapewc3/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
x
gradients/Conv2D_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"             
�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterdropout_1/mulgradients/Conv2D_2_grad/Shape_11gradients/BiasAdd_2_grad/tuple/control_dependency*
data_formatNHWC*&
_output_shapes
: *
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*&
_output_shapes
: *?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter
o
"gradients/dropout_1/mul_grad/ShapeShapedropout_1/Div*
T0*
out_type0*
_output_shapes
:
s
$gradients/dropout_1/mul_grad/Shape_1Shapedropout_1/Floor*
T0*
out_type0*
_output_shapes
:
�
2gradients/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_1/mul_grad/Shape$gradients/dropout_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
 gradients/dropout_1/mul_grad/mulMul0gradients/Conv2D_2_grad/tuple/control_dependencydropout_1/Floor*
T0*/
_output_shapes
:���������
�
 gradients/dropout_1/mul_grad/SumSum gradients/dropout_1/mul_grad/mul2gradients/dropout_1/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
$gradients/dropout_1/mul_grad/ReshapeReshape gradients/dropout_1/mul_grad/Sum"gradients/dropout_1/mul_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
�
"gradients/dropout_1/mul_grad/mul_1Muldropout_1/Div0gradients/Conv2D_2_grad/tuple/control_dependency*
T0*/
_output_shapes
:���������
�
"gradients/dropout_1/mul_grad/Sum_1Sum"gradients/dropout_1/mul_grad/mul_14gradients/dropout_1/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
&gradients/dropout_1/mul_grad/Reshape_1Reshape"gradients/dropout_1/mul_grad/Sum_1$gradients/dropout_1/mul_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:���������
�
-gradients/dropout_1/mul_grad/tuple/group_depsNoOp%^gradients/dropout_1/mul_grad/Reshape'^gradients/dropout_1/mul_grad/Reshape_1
�
5gradients/dropout_1/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_1/mul_grad/Reshape.^gradients/dropout_1/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*7
_class-
+)loc:@gradients/dropout_1/mul_grad/Reshape
�
7gradients/dropout_1/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_1/mul_grad/Reshape_1.^gradients/dropout_1/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*9
_class/
-+loc:@gradients/dropout_1/mul_grad/Reshape_1
k
"gradients/dropout_1/Div_grad/ShapeShape	MaxPool_1*
T0*
out_type0*
_output_shapes
:
g
$gradients/dropout_1/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
2gradients/dropout_1/Div_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_1/Div_grad/Shape$gradients/dropout_1/Div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/dropout_1/Div_grad/truedivDiv5gradients/dropout_1/mul_grad/tuple/control_dependencystrided_slice_1*
T0*/
_output_shapes
:���������
�
 gradients/dropout_1/Div_grad/SumSum$gradients/dropout_1/Div_grad/truediv2gradients/dropout_1/Div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
$gradients/dropout_1/Div_grad/ReshapeReshape gradients/dropout_1/Div_grad/Sum"gradients/dropout_1/Div_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
l
 gradients/dropout_1/Div_grad/NegNeg	MaxPool_1*
T0*/
_output_shapes
:���������
_
#gradients/dropout_1/Div_grad/SquareSquarestrided_slice_1*
T0*
_output_shapes
: 
�
&gradients/dropout_1/Div_grad/truediv_1Div gradients/dropout_1/Div_grad/Neg#gradients/dropout_1/Div_grad/Square*
T0*/
_output_shapes
:���������
�
 gradients/dropout_1/Div_grad/mulMul5gradients/dropout_1/mul_grad/tuple/control_dependency&gradients/dropout_1/Div_grad/truediv_1*
T0*/
_output_shapes
:���������
�
"gradients/dropout_1/Div_grad/Sum_1Sum gradients/dropout_1/Div_grad/mul4gradients/dropout_1/Div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
&gradients/dropout_1/Div_grad/Reshape_1Reshape"gradients/dropout_1/Div_grad/Sum_1$gradients/dropout_1/Div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
-gradients/dropout_1/Div_grad/tuple/group_depsNoOp%^gradients/dropout_1/Div_grad/Reshape'^gradients/dropout_1/Div_grad/Reshape_1
�
5gradients/dropout_1/Div_grad/tuple/control_dependencyIdentity$gradients/dropout_1/Div_grad/Reshape.^gradients/dropout_1/Div_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*7
_class-
+)loc:@gradients/dropout_1/Div_grad/Reshape
�
7gradients/dropout_1/Div_grad/tuple/control_dependency_1Identity&gradients/dropout_1/Div_grad/Reshape_1.^gradients/dropout_1/Div_grad/tuple/group_deps*
T0*
_output_shapes
: *9
_class/
-+loc:@gradients/dropout_1/Div_grad/Reshape_1
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_15gradients/dropout_1/Div_grad/tuple/control_dependency*
ksize
*
data_formatNHWC*/
_output_shapes
:���������*
strides
*
T0*
paddingSAME
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*/
_output_shapes
:���������
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*
_output_shapes
:*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad
h
gradients/Conv2D_1_grad/ShapeShapedropout/mul*
T0*
out_type0*
_output_shapes
:
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/Shapewc2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
x
gradients/Conv2D_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterdropout/mulgradients/Conv2D_1_grad/Shape_11gradients/BiasAdd_1_grad/tuple/control_dependency*
data_formatNHWC*&
_output_shapes
:*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*&
_output_shapes
:*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
k
 gradients/dropout/mul_grad/ShapeShapedropout/Div*
T0*
out_type0*
_output_shapes
:
o
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
T0*
out_type0*
_output_shapes
:
�
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/dropout/mul_grad/mulMul0gradients/Conv2D_1_grad/tuple/control_dependencydropout/Floor*
T0*/
_output_shapes
:���������
�
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
�
 gradients/dropout/mul_grad/mul_1Muldropout/Div0gradients/Conv2D_1_grad/tuple/control_dependency*
T0*/
_output_shapes
:���������
�
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:���������

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
�
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape
�
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1
g
 gradients/dropout/Div_grad/ShapeShapeMaxPool*
T0*
out_type0*
_output_shapes
:
e
"gradients/dropout/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
0gradients/dropout/Div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/Div_grad/Shape"gradients/dropout/Div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
"gradients/dropout/Div_grad/truedivDiv3gradients/dropout/mul_grad/tuple/control_dependencystrided_slice*
T0*/
_output_shapes
:���������
�
gradients/dropout/Div_grad/SumSum"gradients/dropout/Div_grad/truediv0gradients/dropout/Div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
"gradients/dropout/Div_grad/ReshapeReshapegradients/dropout/Div_grad/Sum gradients/dropout/Div_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
h
gradients/dropout/Div_grad/NegNegMaxPool*
T0*/
_output_shapes
:���������
[
!gradients/dropout/Div_grad/SquareSquarestrided_slice*
T0*
_output_shapes
: 
�
$gradients/dropout/Div_grad/truediv_1Divgradients/dropout/Div_grad/Neg!gradients/dropout/Div_grad/Square*
T0*/
_output_shapes
:���������
�
gradients/dropout/Div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/Div_grad/truediv_1*
T0*/
_output_shapes
:���������
�
 gradients/dropout/Div_grad/Sum_1Sumgradients/dropout/Div_grad/mul2gradients/dropout/Div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
$gradients/dropout/Div_grad/Reshape_1Reshape gradients/dropout/Div_grad/Sum_1"gradients/dropout/Div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

+gradients/dropout/Div_grad/tuple/group_depsNoOp#^gradients/dropout/Div_grad/Reshape%^gradients/dropout/Div_grad/Reshape_1
�
3gradients/dropout/Div_grad/tuple/control_dependencyIdentity"gradients/dropout/Div_grad/Reshape,^gradients/dropout/Div_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*5
_class+
)'loc:@gradients/dropout/Div_grad/Reshape
�
5gradients/dropout/Div_grad/tuple/control_dependency_1Identity$gradients/dropout/Div_grad/Reshape_1,^gradients/dropout/Div_grad/tuple/group_deps*
T0*
_output_shapes
: *7
_class-
+)loc:@gradients/dropout/Div_grad/Reshape_1
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool3gradients/dropout/Div_grad/tuple/control_dependency*
ksize
*
data_formatNHWC*/
_output_shapes
:���������..*
strides
*
T0*
paddingSAME
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*/
_output_shapes
:���������..
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*/
_output_shapes
:���������..*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad
b
gradients/Conv2D_grad/ShapeShapeReshape*
T0*
out_type0*
_output_shapes
:
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/Shapewc1/read/gradients/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
v
gradients/Conv2D_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/Shape_1/gradients/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*&
_output_shapes
:*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*1
_output_shapes
:�����������*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*&
_output_shapes
:*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter
b
GradientDescent/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *�Q8=
�
/GradientDescent/update_wc1/ApplyGradientDescentApplyGradientDescentwc1GradientDescent/learning_rate0gradients/Conv2D_grad/tuple/control_dependency_1*
T0*
use_locking( *&
_output_shapes
:*
_class

loc:@wc1
�
/GradientDescent/update_wc2/ApplyGradientDescentApplyGradientDescentwc2GradientDescent/learning_rate2gradients/Conv2D_1_grad/tuple/control_dependency_1*
T0*
use_locking( *&
_output_shapes
:*
_class

loc:@wc2
�
/GradientDescent/update_wc3/ApplyGradientDescentApplyGradientDescentwc3GradientDescent/learning_rate2gradients/Conv2D_2_grad/tuple/control_dependency_1*
T0*
use_locking( *&
_output_shapes
: *
_class

loc:@wc3
�
/GradientDescent/update_wd1/ApplyGradientDescentApplyGradientDescentwd1GradientDescent/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( * 
_output_shapes
:
�	�*
_class

loc:@wd1
�
1GradientDescent/update_w_out/ApplyGradientDescentApplyGradientDescentw_outGradientDescent/learning_rate2gradients/MatMul_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes
:	�*
_class

loc:@w_out
�
/GradientDescent/update_bc1/ApplyGradientDescentApplyGradientDescentbc1GradientDescent/learning_rate1gradients/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes
:*
_class

loc:@bc1
�
/GradientDescent/update_bc2/ApplyGradientDescentApplyGradientDescentbc2GradientDescent/learning_rate3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes
:*
_class

loc:@bc2
�
/GradientDescent/update_bc3/ApplyGradientDescentApplyGradientDescentbc3GradientDescent/learning_rate3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes
: *
_class

loc:@bc3
�
/GradientDescent/update_bd1/ApplyGradientDescentApplyGradientDescentbd1GradientDescent/learning_rate-gradients/Add_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes	
:�*
_class

loc:@bd1
�
1GradientDescent/update_b_out/ApplyGradientDescentApplyGradientDescentb_outGradientDescent/learning_rate/gradients/Add_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes
:*
_class

loc:@b_out
�
GradientDescentNoOp0^GradientDescent/update_wc1/ApplyGradientDescent0^GradientDescent/update_wc2/ApplyGradientDescent0^GradientDescent/update_wc3/ApplyGradientDescent0^GradientDescent/update_wd1/ApplyGradientDescent2^GradientDescent/update_w_out/ApplyGradientDescent0^GradientDescent/update_bc1/ApplyGradientDescent0^GradientDescent/update_bc2/ApplyGradientDescent0^GradientDescent/update_bc3/ApplyGradientDescent0^GradientDescent/update_bd1/ApplyGradientDescent2^GradientDescent/update_b_out/ApplyGradientDescent
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
b
ArgMaxArgMaxpredArgMax/dimension*
T0*

Tidx0*#
_output_shapes
:���������
T
ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
c
ArgMax_1ArgMaxyArgMax_1/dimension*
T0*

Tidx0*#
_output_shapes
:���������
T
ArgMax_2/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
f
ArgMax_2ArgMaxpredArgMax_2/dimension*
T0*

Tidx0*#
_output_shapes
:���������
T
ArgMax_3/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
c
ArgMax_3ArgMaxyArgMax_3/dimension*
T0*

Tidx0*#
_output_shapes
:���������
P
EqualEqualArgMax_2ArgMax_3*
T0	*#
_output_shapes
:���������
R
Cast_1CastEqual*

SrcT0
*#
_output_shapes
:���������*

DstT0
Q
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
_
accuracyMeanCast_1Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
[
ScalarSummary/tagsConst*
dtype0*
_output_shapes
: *
valueB BAccuracy
]
ScalarSummaryScalarSummaryScalarSummary/tagsaccuracy*
T0*
_output_shapes
: 
Y
ScalarSummary_1/tagsConst*
dtype0*
_output_shapes
: *
valueB
 BLoss
]
ScalarSummary_1ScalarSummaryScalarSummary_1/tagscost*
T0*
_output_shapes
: 
h
Reshape_5/shapeConst*
dtype0*
_output_shapes
:*%
valueB"�����   �      
r
	Reshape_5ReshapexReshape_5/shape*
T0*
Tshape0*1
_output_shapes
:�����������
Y
ImageSummary/tagConst*
dtype0*
_output_shapes
: *
valueB BOriginal
�
ImageSummaryImageSummaryImageSummary/tag	Reshape_5*
T0*
	bad_colorB:�  �*
_output_shapes
: *

max_images
h
Reshape_6/shapeConst*
dtype0*
_output_shapes
:*%
valueB"�����   �      
r
	Reshape_6ReshapexReshape_6/shape*
T0*
Tshape0*1
_output_shapes
:�����������
�
Conv2D_3Conv2D	Reshape_6wc1/read*
data_formatNHWC*/
_output_shapes
:���������..*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
y
	BiasAdd_3BiasAddConv2D_3bc1/read*
T0*
data_formatNHWC*/
_output_shapes
:���������..
S
Relu_4Relu	BiasAdd_3*
T0*/
_output_shapes
:���������..
L
conv1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
W
conv1AddRelu_4conv1/y*
T0*/
_output_shapes
:���������..
h
Reshape_7/shapeConst*
dtype0*
_output_shapes
:*%
valueB"����.   .      
t
	Reshape_7Reshapeconv1Reshape_7/shape*
T0*
Tshape0*/
_output_shapes
:���������..
Y
ImageSummary_1/tagConst*
dtype0*
_output_shapes
: *
valueB B1.Conv
�
ImageSummary_1ImageSummaryImageSummary_1/tag	Reshape_7*
T0*
	bad_colorB:�  �*
_output_shapes
: *

max_images
�
Conv2D_4Conv2Dconv1wc2/read*
data_formatNHWC*/
_output_shapes
:���������..*
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
y
	BiasAdd_4BiasAddConv2D_4bc2/read*
T0*
data_formatNHWC*/
_output_shapes
:���������..
S
Relu_5Relu	BiasAdd_4*
T0*/
_output_shapes
:���������..
L
conv2/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
W
conv2AddRelu_5conv2/y*
T0*/
_output_shapes
:���������..
h
Reshape_8/shapeConst*
dtype0*
_output_shapes
:*%
valueB"����.   .      
t
	Reshape_8Reshapeconv2Reshape_8/shape*
T0*
Tshape0*/
_output_shapes
:���������..
Y
ImageSummary_2/tagConst*
dtype0*
_output_shapes
: *
valueB B2.Conv
�
ImageSummary_2ImageSummaryImageSummary_2/tag	Reshape_8*
T0*
	bad_colorB:�  �*
_output_shapes
: *

max_images
�
Conv2D_5Conv2Dconv2wc3/read*
data_formatNHWC*/
_output_shapes
:���������.. *
strides
*
T0*
paddingSAME*
use_cudnn_on_gpu(
y
	BiasAdd_5BiasAddConv2D_5bc3/read*
T0*
data_formatNHWC*/
_output_shapes
:���������.. 
S
Relu_6Relu	BiasAdd_5*
T0*/
_output_shapes
:���������.. 
L
conv3/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
W
conv3AddRelu_6conv3/y*
T0*/
_output_shapes
:���������.. 
h
Reshape_9/shapeConst*
dtype0*
_output_shapes
:*%
valueB"����.   .      
t
	Reshape_9Reshapeconv3Reshape_9/shape*
T0*
Tshape0*/
_output_shapes
:���������..
Y
ImageSummary_3/tagConst*
dtype0*
_output_shapes
: *
valueB B3.Conv
�
ImageSummary_3ImageSummaryImageSummary_3/tag	Reshape_9*
T0*
	bad_colorB:�  �*
_output_shapes
: *

max_images 
e
HistogramSummary/tagConst*
dtype0*
_output_shapes
: *!
valueB BHistogram 1.Conv
e
HistogramSummaryHistogramSummaryHistogramSummary/tagwc1/read*
T0*
_output_shapes
: 
e
HistogramSummary_1/tagConst*
dtype0*
_output_shapes
: *
valueB BHistogram pred
e
HistogramSummary_1HistogramSummaryHistogramSummary_1/tagpred*
T0*
_output_shapes
: 
�
MergeSummary/MergeSummaryMergeSummaryScalarSummaryScalarSummary_1ImageSummaryImageSummary_1ImageSummary_2ImageSummary_3HistogramSummaryHistogramSummary_1*
N*
_output_shapes
: 
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
�
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:
*I
value@B>
Bb_outBbc1Bbc2Bbc3Bbd1Bw_outBwc1Bwc2Bwc3Bwd1
w
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:
*'
valueB
B B B B B B B B B B 
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesb_outbc1bc2bc3bd1w_outwc1wc2wc3wd1*
dtypes
2

}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_output_shapes
: *
_class
loc:@save/Const
i
save/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBb_out
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/AssignAssignb_outsave/RestoreV2*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*
_class

loc:@b_out
i
save/RestoreV2_1/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBbc1
j
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_1Assignbc1save/RestoreV2_1*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*
_class

loc:@bc1
i
save/RestoreV2_2/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBbc2
j
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_2Assignbc2save/RestoreV2_2*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*
_class

loc:@bc2
i
save/RestoreV2_3/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBbc3
j
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3Assignbc3save/RestoreV2_3*
T0*
use_locking(*
validate_shape(*
_output_shapes
: *
_class

loc:@bc3
i
save/RestoreV2_4/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBbd1
j
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_4Assignbd1save/RestoreV2_4*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:�*
_class

loc:@bd1
k
save/RestoreV2_5/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBw_out
j
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_5Assignw_outsave/RestoreV2_5*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	�*
_class

loc:@w_out
i
save/RestoreV2_6/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBwc1
j
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_6Assignwc1save/RestoreV2_6*
T0*
use_locking(*
validate_shape(*&
_output_shapes
:*
_class

loc:@wc1
i
save/RestoreV2_7/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBwc2
j
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_7Assignwc2save/RestoreV2_7*
T0*
use_locking(*
validate_shape(*&
_output_shapes
:*
_class

loc:@wc2
i
save/RestoreV2_8/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBwc3
j
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_8Assignwc3save/RestoreV2_8*
T0*
use_locking(*
validate_shape(*&
_output_shapes
: *
_class

loc:@wc3
i
save/RestoreV2_9/tensor_namesConst*
dtype0*
_output_shapes
:*
valueBBwd1
j
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_9Assignwd1save/RestoreV2_9*
T0*
use_locking(*
vali
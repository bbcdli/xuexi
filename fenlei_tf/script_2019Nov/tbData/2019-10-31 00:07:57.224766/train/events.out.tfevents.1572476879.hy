       ЃK"	  РѓnзAbrain.Event:2ЉЙЋШ,     @Ф	д-иѓnзA"Лй
Y
xPlaceholder*
dtype0*-
_output_shapes
:џџџџџџџџџР*
shape: 
S
yPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape: 
R
	keep_probPlaceholder*
dtype0*
_output_shapes
:*
shape:
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
 *ЭЬЬ=
Љ
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
dtype0*
seed2*
seedБџх)*
T0*&
_output_shapes
:

random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*&
_output_shapes
:*
T0
l
random_normalAddrandom_normal/mulrandom_normal/mean*&
_output_shapes
:*
T0

wc1Variable*
dtype0*
shared_name *
shape:*
	container *&
_output_shapes
:


wc1/AssignAssignwc1random_normal*
use_locking(*&
_output_shapes
:*
validate_shape(*
_class

loc:@wc1*
T0
b
wc1/readIdentitywc1*&
_output_shapes
:*
_class

loc:@wc1*
T0
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
 *ЭЬЬ=
­
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
dtype0*
seed2*
seedБџх)*
T0*&
_output_shapes
:

random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*&
_output_shapes
:*
T0
r
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*&
_output_shapes
:*
T0

wc2Variable*
dtype0*
shared_name *
shape:*
	container *&
_output_shapes
:


wc2/AssignAssignwc2random_normal_1*
use_locking(*&
_output_shapes
:*
validate_shape(*
_class

loc:@wc2*
T0
b
wc2/readIdentitywc2*&
_output_shapes
:*
_class

loc:@wc2*
T0
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
 *ЭЬЬ=
­
$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*
dtype0*
seed2*
seedБџх)*
T0*&
_output_shapes
: 

random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*&
_output_shapes
: *
T0
r
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*&
_output_shapes
: *
T0

wc3Variable*
dtype0*
shared_name *
shape: *
	container *&
_output_shapes
: 


wc3/AssignAssignwc3random_normal_2*
use_locking(*&
_output_shapes
: *
validate_shape(*
_class

loc:@wc3*
T0
b
wc3/readIdentitywc3*&
_output_shapes
: *
_class

loc:@wc3*
T0
f
random_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"  h  
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
 *ЭЬЬ=
Ї
$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*
dtype0*
seed2*
seedБџх)*
T0* 
_output_shapes
:
	ш

random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev* 
_output_shapes
:
	ш*
T0
l
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean* 
_output_shapes
:
	ш*
T0
y
wd1Variable*
dtype0*
shared_name *
shape:
	ш*
	container * 
_output_shapes
:
	ш


wd1/AssignAssignwd1random_normal_3*
use_locking(* 
_output_shapes
:
	ш*
validate_shape(*
_class

loc:@wd1*
T0
\
wd1/readIdentitywd1* 
_output_shapes
:
	ш*
_class

loc:@wd1*
T0
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
 *ЭЬЬ=
І
$random_normal_4/RandomStandardNormalRandomStandardNormalrandom_normal_4/shape*
dtype0*
seed2*
seedБџх)*
T0*
_output_shapes
:	ш

random_normal_4/mulMul$random_normal_4/RandomStandardNormalrandom_normal_4/stddev*
_output_shapes
:	ш*
T0
k
random_normal_4Addrandom_normal_4/mulrandom_normal_4/mean*
_output_shapes
:	ш*
T0
y
w_outVariable*
dtype0*
shared_name *
shape:	ш*
	container *
_output_shapes
:	ш

w_out/AssignAssignw_outrandom_normal_4*
use_locking(*
_output_shapes
:	ш*
validate_shape(*
_class

loc:@w_out*
T0
a

w_out/readIdentityw_out*
_output_shapes
:	ш*
_class

loc:@w_out*
T0
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
 *  ?

$random_normal_5/RandomStandardNormalRandomStandardNormalrandom_normal_5/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:
}
random_normal_5/mulMul$random_normal_5/RandomStandardNormalrandom_normal_5/stddev*
_output_shapes
:*
T0
f
random_normal_5Addrandom_normal_5/mulrandom_normal_5/mean*
_output_shapes
:*
T0
m
bc1Variable*
dtype0*
shared_name *
shape:*
	container *
_output_shapes
:


bc1/AssignAssignbc1random_normal_5*
use_locking(*
_output_shapes
:*
validate_shape(*
_class

loc:@bc1*
T0
V
bc1/readIdentitybc1*
_output_shapes
:*
_class

loc:@bc1*
T0
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
 *  ?

$random_normal_6/RandomStandardNormalRandomStandardNormalrandom_normal_6/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:
}
random_normal_6/mulMul$random_normal_6/RandomStandardNormalrandom_normal_6/stddev*
_output_shapes
:*
T0
f
random_normal_6Addrandom_normal_6/mulrandom_normal_6/mean*
_output_shapes
:*
T0
m
bc2Variable*
dtype0*
shared_name *
shape:*
	container *
_output_shapes
:


bc2/AssignAssignbc2random_normal_6*
use_locking(*
_output_shapes
:*
validate_shape(*
_class

loc:@bc2*
T0
V
bc2/readIdentitybc2*
_output_shapes
:*
_class

loc:@bc2*
T0
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
 *  ?

$random_normal_7/RandomStandardNormalRandomStandardNormalrandom_normal_7/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
: 
}
random_normal_7/mulMul$random_normal_7/RandomStandardNormalrandom_normal_7/stddev*
_output_shapes
: *
T0
f
random_normal_7Addrandom_normal_7/mulrandom_normal_7/mean*
_output_shapes
: *
T0
m
bc3Variable*
dtype0*
shared_name *
shape: *
	container *
_output_shapes
: 


bc3/AssignAssignbc3random_normal_7*
use_locking(*
_output_shapes
: *
validate_shape(*
_class

loc:@bc3*
T0
V
bc3/readIdentitybc3*
_output_shapes
: *
_class

loc:@bc3*
T0
`
random_normal_8/shapeConst*
dtype0*
_output_shapes
:*
valueB:ш
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
 *  ?

$random_normal_8/RandomStandardNormalRandomStandardNormalrandom_normal_8/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes	
:ш
~
random_normal_8/mulMul$random_normal_8/RandomStandardNormalrandom_normal_8/stddev*
_output_shapes	
:ш*
T0
g
random_normal_8Addrandom_normal_8/mulrandom_normal_8/mean*
_output_shapes	
:ш*
T0
o
bd1Variable*
dtype0*
shared_name *
shape:ш*
	container *
_output_shapes	
:ш


bd1/AssignAssignbd1random_normal_8*
use_locking(*
_output_shapes	
:ш*
validate_shape(*
_class

loc:@bd1*
T0
W
bd1/readIdentitybd1*
_output_shapes	
:ш*
_class

loc:@bd1*
T0
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
 *  ?

$random_normal_9/RandomStandardNormalRandomStandardNormalrandom_normal_9/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:
}
random_normal_9/mulMul$random_normal_9/RandomStandardNormalrandom_normal_9/stddev*
_output_shapes
:*
T0
f
random_normal_9Addrandom_normal_9/mulrandom_normal_9/mean*
_output_shapes
:*
T0
o
b_outVariable*
dtype0*
shared_name *
shape:*
	container *
_output_shapes
:

b_out/AssignAssignb_outrandom_normal_9*
use_locking(*
_output_shapes
:*
validate_shape(*
_class

loc:@b_out*
T0
\

b_out/readIdentityb_out*
_output_shapes
:*
_class

loc:@b_out*
T0
f
Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"џџџџИ   И      
n
ReshapeReshapexReshape/shape*1
_output_shapes
:џџџџџџџџџИИ*
T0*
Tshape0
Д
Conv2DConv2DReshapewc1/read*
paddingSAME*
data_formatNHWC*
strides
*/
_output_shapes
:џџџџџџџџџ..*
T0*
use_cudnn_on_gpu(
u
BiasAddBiasAddConv2Dbc1/read*/
_output_shapes
:џџџџџџџџџ..*
T0*
data_formatNHWC
O
ReluReluBiasAdd*/
_output_shapes
:џџџџџџџџџ..*
T0
Є
MaxPoolMaxPoolRelu*
paddingSAME*
data_formatNHWC*
strides
*/
_output_shapes
:џџџџџџџџџ*
ksize
*
T0
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
§
strided_sliceStridedSlice	keep_probstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
new_axis_mask *

begin_mask *
Index0*
ellipsis_mask *
_output_shapes
: *
shrink_axis_mask*
end_mask *
T0
T
dropout/ShapeShapeMaxPool*
out_type0*
T0*
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
 *  ?
Є
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
dtype0*
seed2 *

seed *
T0*/
_output_shapes
:џџџџџџџџџ
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0

dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*/
_output_shapes
:џџџџџџџџџ*
T0

dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*/
_output_shapes
:џџџџџџџџџ*
T0
s
dropout/addAddstrided_slicedropout/random_uniform*/
_output_shapes
:џџџџџџџџџ*
T0
]
dropout/FloorFloordropout/add*/
_output_shapes
:џџџџџџџџџ*
T0
d
dropout/DivDivMaxPoolstrided_slice*/
_output_shapes
:џџџџџџџџџ*
T0
h
dropout/mulMuldropout/Divdropout/Floor*/
_output_shapes
:џџџџџџџџџ*
T0
К
Conv2D_1Conv2Ddropout/mulwc2/read*
paddingSAME*
data_formatNHWC*
strides
*/
_output_shapes
:џџџџџџџџџ*
T0*
use_cudnn_on_gpu(
y
	BiasAdd_1BiasAddConv2D_1bc2/read*/
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
S
Relu_1Relu	BiasAdd_1*/
_output_shapes
:џџџџџџџџџ*
T0
Ј
	MaxPool_1MaxPoolRelu_1*
paddingSAME*
data_formatNHWC*
strides
*/
_output_shapes
:џџџџџџџџџ*
ksize
*
T0
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

strided_slice_1StridedSlice	keep_probstrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
new_axis_mask *

begin_mask *
Index0*
ellipsis_mask *
_output_shapes
: *
shrink_axis_mask*
end_mask *
T0
X
dropout_1/ShapeShape	MaxPool_1*
out_type0*
T0*
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
 *  ?
Ј
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape*
dtype0*
seed2 *

seed *
T0*/
_output_shapes
:џџџџџџџџџ

dropout_1/random_uniform/subSubdropout_1/random_uniform/maxdropout_1/random_uniform/min*
_output_shapes
: *
T0
Ѓ
dropout_1/random_uniform/mulMul&dropout_1/random_uniform/RandomUniformdropout_1/random_uniform/sub*/
_output_shapes
:џџџџџџџџџ*
T0

dropout_1/random_uniformAdddropout_1/random_uniform/muldropout_1/random_uniform/min*/
_output_shapes
:џџџџџџџџџ*
T0
y
dropout_1/addAddstrided_slice_1dropout_1/random_uniform*/
_output_shapes
:џџџџџџџџџ*
T0
a
dropout_1/FloorFloordropout_1/add*/
_output_shapes
:џџџџџџџџџ*
T0
j
dropout_1/DivDiv	MaxPool_1strided_slice_1*/
_output_shapes
:џџџџџџџџџ*
T0
n
dropout_1/mulMuldropout_1/Divdropout_1/Floor*/
_output_shapes
:џџџџџџџџџ*
T0
М
Conv2D_2Conv2Ddropout_1/mulwc3/read*
paddingSAME*
data_formatNHWC*
strides
*/
_output_shapes
:џџџџџџџџџ *
T0*
use_cudnn_on_gpu(
y
	BiasAdd_2BiasAddConv2D_2bc3/read*/
_output_shapes
:џџџџџџџџџ *
T0*
data_formatNHWC
S
Relu_2Relu	BiasAdd_2*/
_output_shapes
:џџџџџџџџџ *
T0
Ј
	MaxPool_2MaxPoolRelu_2*
paddingSAME*
data_formatNHWC*
strides
*/
_output_shapes
:џџџџџџџџџ *
ksize
*
T0
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

strided_slice_2StridedSlice	keep_probstrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
new_axis_mask *

begin_mask *
Index0*
ellipsis_mask *
_output_shapes
: *
shrink_axis_mask*
end_mask *
T0
X
dropout_2/ShapeShape	MaxPool_2*
out_type0*
T0*
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
 *  ?
Ј
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape*
dtype0*
seed2 *

seed *
T0*/
_output_shapes
:џџџџџџџџџ 

dropout_2/random_uniform/subSubdropout_2/random_uniform/maxdropout_2/random_uniform/min*
_output_shapes
: *
T0
Ѓ
dropout_2/random_uniform/mulMul&dropout_2/random_uniform/RandomUniformdropout_2/random_uniform/sub*/
_output_shapes
:џџџџџџџџџ *
T0

dropout_2/random_uniformAdddropout_2/random_uniform/muldropout_2/random_uniform/min*/
_output_shapes
:џџџџџџџџџ *
T0
y
dropout_2/addAddstrided_slice_2dropout_2/random_uniform*/
_output_shapes
:џџџџџџџџџ *
T0
a
dropout_2/FloorFloordropout_2/add*/
_output_shapes
:џџџџџџџџџ *
T0
j
dropout_2/DivDiv	MaxPool_2strided_slice_2*/
_output_shapes
:џџџџџџџџџ *
T0
n
dropout_2/mulMuldropout_2/Divdropout_2/Floor*/
_output_shapes
:џџџџџџџџџ *
T0
`
Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ  
u
	Reshape_1Reshapedropout_2/mulReshape_1/shape*(
_output_shapes
:џџџџџџџџџ	*
T0*
Tshape0
~
MatMulMatMul	Reshape_1wd1/read*(
_output_shapes
:џџџџџџџџџш*
transpose_a( *
transpose_b( *
T0
O
AddAddMatMulbd1/read*(
_output_shapes
:џџџџџџџџџш*
T0
F
Relu_3ReluAdd*(
_output_shapes
:џџџџџџџџџш*
T0
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

strided_slice_3StridedSlice	keep_probstrided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_2*
new_axis_mask *

begin_mask *
Index0*
ellipsis_mask *
_output_shapes
: *
shrink_axis_mask*
end_mask *
T0
U
dropout_3/ShapeShapeRelu_3*
out_type0*
T0*
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
 *  ?
Ё
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:џџџџџџџџџш

dropout_3/random_uniform/subSubdropout_3/random_uniform/maxdropout_3/random_uniform/min*
_output_shapes
: *
T0

dropout_3/random_uniform/mulMul&dropout_3/random_uniform/RandomUniformdropout_3/random_uniform/sub*(
_output_shapes
:џџџџџџџџџш*
T0

dropout_3/random_uniformAdddropout_3/random_uniform/muldropout_3/random_uniform/min*(
_output_shapes
:џџџџџџџџџш*
T0
r
dropout_3/addAddstrided_slice_3dropout_3/random_uniform*(
_output_shapes
:џџџџџџџџџш*
T0
Z
dropout_3/FloorFloordropout_3/add*(
_output_shapes
:џџџџџџџџџш*
T0
`
dropout_3/DivDivRelu_3strided_slice_3*(
_output_shapes
:џџџџџџџџџш*
T0
g
dropout_3/mulMuldropout_3/Divdropout_3/Floor*(
_output_shapes
:џџџџџџџџџш*
T0

MatMul_1MatMuldropout_3/mul
w_out/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
T
Add_1AddMatMul_1
b_out/read*'
_output_shapes
:џџџџџџџџџ*
T0
K
pred/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
L
predAddAdd_1pred/y*'
_output_shapes
:џџџџџџџџџ*
T0
F
RankConst*
dtype0*
_output_shapes
: *
value	B :
I
ShapeShapepred*
out_type0*
T0*
_output_shapes
:
H
Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
K
Shape_1Shapepred*
out_type0*
T0*
_output_shapes
:
G
Sub/yConst*
dtype0*
_output_shapes
: *
value	B :
:
SubSubRank_1Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*
_output_shapes
:*
N*

axis *
T0
T

Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
b
SliceSliceShape_1Slice/begin
Slice/size*
_output_shapes
:*
Index0*
T0
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
џџџџџџџџџ
i
concatConcatconcat/concat_dimconcat/values_0Slice*
_output_shapes
:*
N*
T0
k
	Reshape_2Reshapepredconcat*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*
Tshape0
H
Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
H
Shape_2Shapey*
out_type0*
T0*
_output_shapes
:
I
Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
>
Sub_1SubRank_2Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*
_output_shapes
:*
N*

axis *
T0
V
Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
_output_shapes
:*
Index0*
T0
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
џџџџџџџџџ
q
concat_1Concatconcat_1/concat_dimconcat_1/values_0Slice_1*
_output_shapes
:*
N*
T0
j
	Reshape_3Reshapeyconcat_1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*
Tshape0

SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ*
T0
I
Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB: 
U
Slice_2/sizePackSub_2*
_output_shapes
:*
N*

axis *
T0
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*#
_output_shapes
:џџџџџџџџџ*
Index0*
T0
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
O
ConstConst*
dtype0*
_output_shapes
:*
valueB: 
\
costMean	Reshape_4Const*
_output_shapes
: *
	keep_dims( *
T0*

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
 *  ?
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
k
!gradients/cost_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:

gradients/cost_grad/ReshapeReshapegradients/Fill!gradients/cost_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
b
gradients/cost_grad/ShapeShape	Reshape_4*
out_type0*
T0*
_output_shapes
:

gradients/cost_grad/TileTilegradients/cost_grad/Reshapegradients/cost_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:џџџџџџџџџ
d
gradients/cost_grad/Shape_1Shape	Reshape_4*
out_type0*
T0*
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

gradients/cost_grad/ProdProdgradients/cost_grad/Shape_1gradients/cost_grad/Const*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
e
gradients/cost_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 

gradients/cost_grad/Prod_1Prodgradients/cost_grad/Shape_2gradients/cost_grad/Const_1*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
_
gradients/cost_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :

gradients/cost_grad/MaximumMaximumgradients/cost_grad/Prod_1gradients/cost_grad/Maximum/y*
_output_shapes
: *
T0
{
gradients/cost_grad/floordivDivgradients/cost_grad/Prodgradients/cost_grad/Maximum*
_output_shapes
: *
T0
n
gradients/cost_grad/CastCastgradients/cost_grad/floordiv*
_output_shapes
: *

SrcT0*

DstT0

gradients/cost_grad/truedivDivgradients/cost_grad/Tilegradients/cost_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
out_type0*
T0*
_output_shapes
:
Є
 gradients/Reshape_4_grad/ReshapeReshapegradients/cost_grad/truedivgradients/Reshape_4_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0

;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
т
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
Ь
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
b
gradients/Reshape_2_grad/ShapeShapepred*
out_type0*
T0*
_output_shapes
:
Н
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
^
gradients/pred_grad/ShapeShapeAdd_1*
out_type0*
T0*
_output_shapes
:
^
gradients/pred_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
З
)gradients/pred_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pred_grad/Shapegradients/pred_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/pred_grad/SumSum gradients/Reshape_2_grad/Reshape)gradients/pred_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0

gradients/pred_grad/ReshapeReshapegradients/pred_grad/Sumgradients/pred_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Џ
gradients/pred_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape+gradients/pred_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0

gradients/pred_grad/Reshape_1Reshapegradients/pred_grad/Sum_1gradients/pred_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
j
$gradients/pred_grad/tuple/group_depsNoOp^gradients/pred_grad/Reshape^gradients/pred_grad/Reshape_1
о
,gradients/pred_grad/tuple/control_dependencyIdentitygradients/pred_grad/Reshape%^gradients/pred_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*.
_class$
" loc:@gradients/pred_grad/Reshape*
T0
г
.gradients/pred_grad/tuple/control_dependency_1Identitygradients/pred_grad/Reshape_1%^gradients/pred_grad/tuple/group_deps*
_output_shapes
: *0
_class&
$"loc:@gradients/pred_grad/Reshape_1*
T0
b
gradients/Add_1_grad/ShapeShapeMatMul_1*
out_type0*
T0*
_output_shapes
:
f
gradients/Add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
К
*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Й
gradients/Add_1_grad/SumSum,gradients/pred_grad/tuple/control_dependency*gradients/Add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0

gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Н
gradients/Add_1_grad/Sum_1Sum,gradients/pred_grad/tuple/control_dependency,gradients/Add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0

gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
т
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*/
_class%
#!loc:@gradients/Add_1_grad/Reshape*
T0
л
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*
_output_shapes
:*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1*
T0
М
gradients/MatMul_1_grad/MatMulMatMul-gradients/Add_1_grad/tuple/control_dependency
w_out/read*(
_output_shapes
:џџџџџџџџџш*
transpose_a( *
transpose_b(*
T0
И
 gradients/MatMul_1_grad/MatMul_1MatMuldropout_3/mul-gradients/Add_1_grad/tuple/control_dependency*
_output_shapes
:	ш*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
э
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџш*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0
ъ
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
_output_shapes
:	ш*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0
o
"gradients/dropout_3/mul_grad/ShapeShapedropout_3/Div*
out_type0*
T0*
_output_shapes
:
s
$gradients/dropout_3/mul_grad/Shape_1Shapedropout_3/Floor*
out_type0*
T0*
_output_shapes
:
в
2gradients/dropout_3/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_3/mul_grad/Shape$gradients/dropout_3/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

 gradients/dropout_3/mul_grad/mulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout_3/Floor*(
_output_shapes
:џџџџџџџџџш*
T0
Н
 gradients/dropout_3/mul_grad/SumSum gradients/dropout_3/mul_grad/mul2gradients/dropout_3/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
Ж
$gradients/dropout_3/mul_grad/ReshapeReshape gradients/dropout_3/mul_grad/Sum"gradients/dropout_3/mul_grad/Shape*(
_output_shapes
:џџџџџџџџџш*
T0*
Tshape0

"gradients/dropout_3/mul_grad/mul_1Muldropout_3/Div0gradients/MatMul_1_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџш*
T0
У
"gradients/dropout_3/mul_grad/Sum_1Sum"gradients/dropout_3/mul_grad/mul_14gradients/dropout_3/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
М
&gradients/dropout_3/mul_grad/Reshape_1Reshape"gradients/dropout_3/mul_grad/Sum_1$gradients/dropout_3/mul_grad/Shape_1*(
_output_shapes
:џџџџџџџџџш*
T0*
Tshape0

-gradients/dropout_3/mul_grad/tuple/group_depsNoOp%^gradients/dropout_3/mul_grad/Reshape'^gradients/dropout_3/mul_grad/Reshape_1

5gradients/dropout_3/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_3/mul_grad/Reshape.^gradients/dropout_3/mul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџш*7
_class-
+)loc:@gradients/dropout_3/mul_grad/Reshape*
T0

7gradients/dropout_3/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_3/mul_grad/Reshape_1.^gradients/dropout_3/mul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџш*9
_class/
-+loc:@gradients/dropout_3/mul_grad/Reshape_1*
T0
h
"gradients/dropout_3/Div_grad/ShapeShapeRelu_3*
out_type0*
T0*
_output_shapes
:
g
$gradients/dropout_3/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
в
2gradients/dropout_3/Div_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_3/Div_grad/Shape$gradients/dropout_3/Div_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
І
$gradients/dropout_3/Div_grad/truedivDiv5gradients/dropout_3/mul_grad/tuple/control_dependencystrided_slice_3*(
_output_shapes
:џџџџџџџџџш*
T0
С
 gradients/dropout_3/Div_grad/SumSum$gradients/dropout_3/Div_grad/truediv2gradients/dropout_3/Div_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
Ж
$gradients/dropout_3/Div_grad/ReshapeReshape gradients/dropout_3/Div_grad/Sum"gradients/dropout_3/Div_grad/Shape*(
_output_shapes
:џџџџџџџџџш*
T0*
Tshape0
b
 gradients/dropout_3/Div_grad/NegNegRelu_3*(
_output_shapes
:џџџџџџџџџш*
T0
_
#gradients/dropout_3/Div_grad/SquareSquarestrided_slice_3*
_output_shapes
: *
T0
Ї
&gradients/dropout_3/Div_grad/truediv_1Div gradients/dropout_3/Div_grad/Neg#gradients/dropout_3/Div_grad/Square*(
_output_shapes
:џџџџџџџџџш*
T0
Й
 gradients/dropout_3/Div_grad/mulMul5gradients/dropout_3/mul_grad/tuple/control_dependency&gradients/dropout_3/Div_grad/truediv_1*(
_output_shapes
:џџџџџџџџџш*
T0
С
"gradients/dropout_3/Div_grad/Sum_1Sum gradients/dropout_3/Div_grad/mul4gradients/dropout_3/Div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
Њ
&gradients/dropout_3/Div_grad/Reshape_1Reshape"gradients/dropout_3/Div_grad/Sum_1$gradients/dropout_3/Div_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

-gradients/dropout_3/Div_grad/tuple/group_depsNoOp%^gradients/dropout_3/Div_grad/Reshape'^gradients/dropout_3/Div_grad/Reshape_1

5gradients/dropout_3/Div_grad/tuple/control_dependencyIdentity$gradients/dropout_3/Div_grad/Reshape.^gradients/dropout_3/Div_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџш*7
_class-
+)loc:@gradients/dropout_3/Div_grad/Reshape*
T0
ї
7gradients/dropout_3/Div_grad/tuple/control_dependency_1Identity&gradients/dropout_3/Div_grad/Reshape_1.^gradients/dropout_3/Div_grad/tuple/group_deps*
_output_shapes
: *9
_class/
-+loc:@gradients/dropout_3/Div_grad/Reshape_1*
T0

gradients/Relu_3_grad/ReluGradReluGrad5gradients/dropout_3/Div_grad/tuple/control_dependencyRelu_3*(
_output_shapes
:џџџџџџџџџш*
T0
^
gradients/Add_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
e
gradients/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:ш
Д
(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ї
gradients/Add_grad/SumSumgradients/Relu_3_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0

gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*(
_output_shapes
:џџџџџџџџџш*
T0*
Tshape0
Ћ
gradients/Add_grad/Sum_1Sumgradients/Relu_3_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0

gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
_output_shapes	
:ш*
T0*
Tshape0
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
л
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџш*-
_class#
!loc:@gradients/Add_grad/Reshape*
T0
д
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*
_output_shapes	
:ш*/
_class%
#!loc:@gradients/Add_grad/Reshape_1*
T0
Ж
gradients/MatMul_grad/MatMulMatMul+gradients/Add_grad/tuple/control_dependencywd1/read*(
_output_shapes
:џџџџџџџџџ	*
transpose_a( *
transpose_b(*
T0
Б
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1+gradients/Add_grad/tuple/control_dependency* 
_output_shapes
:
	ш*
transpose_a(*
transpose_b( *
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
х
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ	*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0
у
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps* 
_output_shapes
:
	ш*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
k
gradients/Reshape_1_grad/ShapeShapedropout_2/mul*
out_type0*
T0*
_output_shapes
:
У
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*/
_output_shapes
:џџџџџџџџџ *
T0*
Tshape0
o
"gradients/dropout_2/mul_grad/ShapeShapedropout_2/Div*
out_type0*
T0*
_output_shapes
:
s
$gradients/dropout_2/mul_grad/Shape_1Shapedropout_2/Floor*
out_type0*
T0*
_output_shapes
:
в
2gradients/dropout_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_2/mul_grad/Shape$gradients/dropout_2/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

 gradients/dropout_2/mul_grad/mulMul gradients/Reshape_1_grad/Reshapedropout_2/Floor*/
_output_shapes
:џџџџџџџџџ *
T0
Н
 gradients/dropout_2/mul_grad/SumSum gradients/dropout_2/mul_grad/mul2gradients/dropout_2/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
Н
$gradients/dropout_2/mul_grad/ReshapeReshape gradients/dropout_2/mul_grad/Sum"gradients/dropout_2/mul_grad/Shape*/
_output_shapes
:џџџџџџџџџ *
T0*
Tshape0

"gradients/dropout_2/mul_grad/mul_1Muldropout_2/Div gradients/Reshape_1_grad/Reshape*/
_output_shapes
:џџџџџџџџџ *
T0
У
"gradients/dropout_2/mul_grad/Sum_1Sum"gradients/dropout_2/mul_grad/mul_14gradients/dropout_2/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
У
&gradients/dropout_2/mul_grad/Reshape_1Reshape"gradients/dropout_2/mul_grad/Sum_1$gradients/dropout_2/mul_grad/Shape_1*/
_output_shapes
:џџџџџџџџџ *
T0*
Tshape0

-gradients/dropout_2/mul_grad/tuple/group_depsNoOp%^gradients/dropout_2/mul_grad/Reshape'^gradients/dropout_2/mul_grad/Reshape_1

5gradients/dropout_2/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_2/mul_grad/Reshape.^gradients/dropout_2/mul_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ *7
_class-
+)loc:@gradients/dropout_2/mul_grad/Reshape*
T0

7gradients/dropout_2/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_2/mul_grad/Reshape_1.^gradients/dropout_2/mul_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ *9
_class/
-+loc:@gradients/dropout_2/mul_grad/Reshape_1*
T0
k
"gradients/dropout_2/Div_grad/ShapeShape	MaxPool_2*
out_type0*
T0*
_output_shapes
:
g
$gradients/dropout_2/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
в
2gradients/dropout_2/Div_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_2/Div_grad/Shape$gradients/dropout_2/Div_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
­
$gradients/dropout_2/Div_grad/truedivDiv5gradients/dropout_2/mul_grad/tuple/control_dependencystrided_slice_2*/
_output_shapes
:џџџџџџџџџ *
T0
С
 gradients/dropout_2/Div_grad/SumSum$gradients/dropout_2/Div_grad/truediv2gradients/dropout_2/Div_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
Н
$gradients/dropout_2/Div_grad/ReshapeReshape gradients/dropout_2/Div_grad/Sum"gradients/dropout_2/Div_grad/Shape*/
_output_shapes
:џџџџџџџџџ *
T0*
Tshape0
l
 gradients/dropout_2/Div_grad/NegNeg	MaxPool_2*/
_output_shapes
:џџџџџџџџџ *
T0
_
#gradients/dropout_2/Div_grad/SquareSquarestrided_slice_2*
_output_shapes
: *
T0
Ў
&gradients/dropout_2/Div_grad/truediv_1Div gradients/dropout_2/Div_grad/Neg#gradients/dropout_2/Div_grad/Square*/
_output_shapes
:џџџџџџџџџ *
T0
Р
 gradients/dropout_2/Div_grad/mulMul5gradients/dropout_2/mul_grad/tuple/control_dependency&gradients/dropout_2/Div_grad/truediv_1*/
_output_shapes
:џџџџџџџџџ *
T0
С
"gradients/dropout_2/Div_grad/Sum_1Sum gradients/dropout_2/Div_grad/mul4gradients/dropout_2/Div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
Њ
&gradients/dropout_2/Div_grad/Reshape_1Reshape"gradients/dropout_2/Div_grad/Sum_1$gradients/dropout_2/Div_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

-gradients/dropout_2/Div_grad/tuple/group_depsNoOp%^gradients/dropout_2/Div_grad/Reshape'^gradients/dropout_2/Div_grad/Reshape_1

5gradients/dropout_2/Div_grad/tuple/control_dependencyIdentity$gradients/dropout_2/Div_grad/Reshape.^gradients/dropout_2/Div_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ *7
_class-
+)loc:@gradients/dropout_2/Div_grad/Reshape*
T0
ї
7gradients/dropout_2/Div_grad/tuple/control_dependency_1Identity&gradients/dropout_2/Div_grad/Reshape_1.^gradients/dropout_2/Div_grad/tuple/group_deps*
_output_shapes
: *9
_class/
-+loc:@gradients/dropout_2/Div_grad/Reshape_1*
T0

$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_25gradients/dropout_2/Div_grad/tuple/control_dependency*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ *
ksize
*
T0*
data_formatNHWC

gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*/
_output_shapes
:џџџџџџџџџ *
T0

$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
_output_shapes
: *
T0*
data_formatNHWC
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
і
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ *1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*
T0
я
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
_output_shapes
: *7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad*
T0
j
gradients/Conv2D_2_grad/ShapeShapedropout_1/mul*
out_type0*
T0*
_output_shapes
:
Ъ
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/Shapewc3/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
strides
*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
use_cudnn_on_gpu(
x
gradients/Conv2D_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"             
Џ
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterdropout_1/mulgradients/Conv2D_2_grad/Shape_11gradients/BiasAdd_2_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
strides
*&
_output_shapes
: *
T0*
use_cudnn_on_gpu(

(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter

0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*
T0

2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*&
_output_shapes
: *?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*
T0
o
"gradients/dropout_1/mul_grad/ShapeShapedropout_1/Div*
out_type0*
T0*
_output_shapes
:
s
$gradients/dropout_1/mul_grad/Shape_1Shapedropout_1/Floor*
out_type0*
T0*
_output_shapes
:
в
2gradients/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_1/mul_grad/Shape$gradients/dropout_1/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Є
 gradients/dropout_1/mul_grad/mulMul0gradients/Conv2D_2_grad/tuple/control_dependencydropout_1/Floor*/
_output_shapes
:џџџџџџџџџ*
T0
Н
 gradients/dropout_1/mul_grad/SumSum gradients/dropout_1/mul_grad/mul2gradients/dropout_1/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
Н
$gradients/dropout_1/mul_grad/ReshapeReshape gradients/dropout_1/mul_grad/Sum"gradients/dropout_1/mul_grad/Shape*/
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Є
"gradients/dropout_1/mul_grad/mul_1Muldropout_1/Div0gradients/Conv2D_2_grad/tuple/control_dependency*/
_output_shapes
:џџџџџџџџџ*
T0
У
"gradients/dropout_1/mul_grad/Sum_1Sum"gradients/dropout_1/mul_grad/mul_14gradients/dropout_1/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
У
&gradients/dropout_1/mul_grad/Reshape_1Reshape"gradients/dropout_1/mul_grad/Sum_1$gradients/dropout_1/mul_grad/Shape_1*/
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

-gradients/dropout_1/mul_grad/tuple/group_depsNoOp%^gradients/dropout_1/mul_grad/Reshape'^gradients/dropout_1/mul_grad/Reshape_1

5gradients/dropout_1/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_1/mul_grad/Reshape.^gradients/dropout_1/mul_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*7
_class-
+)loc:@gradients/dropout_1/mul_grad/Reshape*
T0

7gradients/dropout_1/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_1/mul_grad/Reshape_1.^gradients/dropout_1/mul_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*9
_class/
-+loc:@gradients/dropout_1/mul_grad/Reshape_1*
T0
k
"gradients/dropout_1/Div_grad/ShapeShape	MaxPool_1*
out_type0*
T0*
_output_shapes
:
g
$gradients/dropout_1/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
в
2gradients/dropout_1/Div_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_1/Div_grad/Shape$gradients/dropout_1/Div_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
­
$gradients/dropout_1/Div_grad/truedivDiv5gradients/dropout_1/mul_grad/tuple/control_dependencystrided_slice_1*/
_output_shapes
:џџџџџџџџџ*
T0
С
 gradients/dropout_1/Div_grad/SumSum$gradients/dropout_1/Div_grad/truediv2gradients/dropout_1/Div_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
Н
$gradients/dropout_1/Div_grad/ReshapeReshape gradients/dropout_1/Div_grad/Sum"gradients/dropout_1/Div_grad/Shape*/
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
l
 gradients/dropout_1/Div_grad/NegNeg	MaxPool_1*/
_output_shapes
:џџџџџџџџџ*
T0
_
#gradients/dropout_1/Div_grad/SquareSquarestrided_slice_1*
_output_shapes
: *
T0
Ў
&gradients/dropout_1/Div_grad/truediv_1Div gradients/dropout_1/Div_grad/Neg#gradients/dropout_1/Div_grad/Square*/
_output_shapes
:џџџџџџџџџ*
T0
Р
 gradients/dropout_1/Div_grad/mulMul5gradients/dropout_1/mul_grad/tuple/control_dependency&gradients/dropout_1/Div_grad/truediv_1*/
_output_shapes
:џџџџџџџџџ*
T0
С
"gradients/dropout_1/Div_grad/Sum_1Sum gradients/dropout_1/Div_grad/mul4gradients/dropout_1/Div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
Њ
&gradients/dropout_1/Div_grad/Reshape_1Reshape"gradients/dropout_1/Div_grad/Sum_1$gradients/dropout_1/Div_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

-gradients/dropout_1/Div_grad/tuple/group_depsNoOp%^gradients/dropout_1/Div_grad/Reshape'^gradients/dropout_1/Div_grad/Reshape_1

5gradients/dropout_1/Div_grad/tuple/control_dependencyIdentity$gradients/dropout_1/Div_grad/Reshape.^gradients/dropout_1/Div_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*7
_class-
+)loc:@gradients/dropout_1/Div_grad/Reshape*
T0
ї
7gradients/dropout_1/Div_grad/tuple/control_dependency_1Identity&gradients/dropout_1/Div_grad/Reshape_1.^gradients/dropout_1/Div_grad/tuple/group_deps*
_output_shapes
: *9
_class/
-+loc:@gradients/dropout_1/Div_grad/Reshape_1*
T0

$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_15gradients/dropout_1/Div_grad/tuple/control_dependency*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ*
ksize
*
T0*
data_formatNHWC

gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*/
_output_shapes
:џџџџџџџџџ*
T0

$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
і
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*
T0
я
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
_output_shapes
:*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
T0
h
gradients/Conv2D_1_grad/ShapeShapedropout/mul*
out_type0*
T0*
_output_shapes
:
Ъ
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/Shapewc2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
strides
*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
use_cudnn_on_gpu(
x
gradients/Conv2D_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            
­
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterdropout/mulgradients/Conv2D_1_grad/Shape_11gradients/BiasAdd_1_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
strides
*&
_output_shapes
:*
T0*
use_cudnn_on_gpu(

(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter

0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*
T0

2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*&
_output_shapes
:*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0
k
 gradients/dropout/mul_grad/ShapeShapedropout/Div*
out_type0*
T0*
_output_shapes
:
o
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
out_type0*
T0*
_output_shapes
:
Ь
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
 
gradients/dropout/mul_grad/mulMul0gradients/Conv2D_1_grad/tuple/control_dependencydropout/Floor*/
_output_shapes
:џџџџџџџџџ*
T0
З
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
З
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*/
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
 
 gradients/dropout/mul_grad/mul_1Muldropout/Div0gradients/Conv2D_1_grad/tuple/control_dependency*/
_output_shapes
:џџџџџџџџџ*
T0
Н
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
Н
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*/
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1

3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*
T0

5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1*
T0
g
 gradients/dropout/Div_grad/ShapeShapeMaxPool*
out_type0*
T0*
_output_shapes
:
e
"gradients/dropout/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
Ь
0gradients/dropout/Div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/Div_grad/Shape"gradients/dropout/Div_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ї
"gradients/dropout/Div_grad/truedivDiv3gradients/dropout/mul_grad/tuple/control_dependencystrided_slice*/
_output_shapes
:џџџџџџџџџ*
T0
Л
gradients/dropout/Div_grad/SumSum"gradients/dropout/Div_grad/truediv0gradients/dropout/Div_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
З
"gradients/dropout/Div_grad/ReshapeReshapegradients/dropout/Div_grad/Sum gradients/dropout/Div_grad/Shape*/
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
h
gradients/dropout/Div_grad/NegNegMaxPool*/
_output_shapes
:џџџџџџџџџ*
T0
[
!gradients/dropout/Div_grad/SquareSquarestrided_slice*
_output_shapes
: *
T0
Ј
$gradients/dropout/Div_grad/truediv_1Divgradients/dropout/Div_grad/Neg!gradients/dropout/Div_grad/Square*/
_output_shapes
:џџџџџџџџџ*
T0
К
gradients/dropout/Div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/Div_grad/truediv_1*/
_output_shapes
:џџџџџџџџџ*
T0
Л
 gradients/dropout/Div_grad/Sum_1Sumgradients/dropout/Div_grad/mul2gradients/dropout/Div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
Є
$gradients/dropout/Div_grad/Reshape_1Reshape gradients/dropout/Div_grad/Sum_1"gradients/dropout/Div_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

+gradients/dropout/Div_grad/tuple/group_depsNoOp#^gradients/dropout/Div_grad/Reshape%^gradients/dropout/Div_grad/Reshape_1

3gradients/dropout/Div_grad/tuple/control_dependencyIdentity"gradients/dropout/Div_grad/Reshape,^gradients/dropout/Div_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*5
_class+
)'loc:@gradients/dropout/Div_grad/Reshape*
T0
я
5gradients/dropout/Div_grad/tuple/control_dependency_1Identity$gradients/dropout/Div_grad/Reshape_1,^gradients/dropout/Div_grad/tuple/group_deps*
_output_shapes
: *7
_class-
+)loc:@gradients/dropout/Div_grad/Reshape_1*
T0

"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool3gradients/dropout/Div_grad/tuple/control_dependency*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ..*
ksize
*
T0*
data_formatNHWC

gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*/
_output_shapes
:џџџџџџџџџ..*
T0

"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
ю
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ..*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*
T0
ч
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
T0
b
gradients/Conv2D_grad/ShapeShapeReshape*
out_type0*
T0*
_output_shapes
:
Ф
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/Shapewc1/read/gradients/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
strides
*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
use_cudnn_on_gpu(
v
gradients/Conv2D_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            
Ѓ
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/Shape_1/gradients/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
strides
*&
_output_shapes
:*
T0*
use_cudnn_on_gpu(

&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter

.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*1
_output_shapes
:џџџџџџџџџИИ*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*
T0

0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*&
_output_shapes
:*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0
b
GradientDescent/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *ьQ8=
љ
/GradientDescent/update_wc1/ApplyGradientDescentApplyGradientDescentwc1GradientDescent/learning_rate0gradients/Conv2D_grad/tuple/control_dependency_1*
use_locking( *&
_output_shapes
:*
_class

loc:@wc1*
T0
ћ
/GradientDescent/update_wc2/ApplyGradientDescentApplyGradientDescentwc2GradientDescent/learning_rate2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_locking( *&
_output_shapes
:*
_class

loc:@wc2*
T0
ћ
/GradientDescent/update_wc3/ApplyGradientDescentApplyGradientDescentwc3GradientDescent/learning_rate2gradients/Conv2D_2_grad/tuple/control_dependency_1*
use_locking( *&
_output_shapes
: *
_class

loc:@wc3*
T0
ѓ
/GradientDescent/update_wd1/ApplyGradientDescentApplyGradientDescentwd1GradientDescent/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( * 
_output_shapes
:
	ш*
_class

loc:@wd1*
T0
њ
1GradientDescent/update_w_out/ApplyGradientDescentApplyGradientDescentw_outGradientDescent/learning_rate2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
_output_shapes
:	ш*
_class

loc:@w_out*
T0
ю
/GradientDescent/update_bc1/ApplyGradientDescentApplyGradientDescentbc1GradientDescent/learning_rate1gradients/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
_output_shapes
:*
_class

loc:@bc1*
T0
№
/GradientDescent/update_bc2/ApplyGradientDescentApplyGradientDescentbc2GradientDescent/learning_rate3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
use_locking( *
_output_shapes
:*
_class

loc:@bc2*
T0
№
/GradientDescent/update_bc3/ApplyGradientDescentApplyGradientDescentbc3GradientDescent/learning_rate3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
use_locking( *
_output_shapes
: *
_class

loc:@bc3*
T0
ы
/GradientDescent/update_bd1/ApplyGradientDescentApplyGradientDescentbd1GradientDescent/learning_rate-gradients/Add_grad/tuple/control_dependency_1*
use_locking( *
_output_shapes	
:ш*
_class

loc:@bd1*
T0
ђ
1GradientDescent/update_b_out/ApplyGradientDescentApplyGradientDescentb_outGradientDescent/learning_rate/gradients/Add_1_grad/tuple/control_dependency_1*
use_locking( *
_output_shapes
:*
_class

loc:@b_out*
T0

GradientDescentNoOp0^GradientDescent/update_wc1/ApplyGradientDescent0^GradientDescent/update_wc2/ApplyGradientDescent0^GradientDescent/update_wc3/ApplyGradientDescent0^GradientDescent/update_wd1/ApplyGradientDescent2^GradientDescent/update_w_out/ApplyGradientDescent0^GradientDescent/update_bc1/ApplyGradientDescent0^GradientDescent/update_bc2/ApplyGradientDescent0^GradientDescent/update_bc3/ApplyGradientDescent0^GradientDescent/update_bd1/ApplyGradientDescent2^GradientDescent/update_b_out/ApplyGradientDescent
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
b
ArgMaxArgMaxpredArgMax/dimension*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
T
ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
c
ArgMax_1ArgMaxyArgMax_1/dimension*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
T
ArgMax_2/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
f
ArgMax_2ArgMaxpredArgMax_2/dimension*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
T
ArgMax_3/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
c
ArgMax_3ArgMaxyArgMax_3/dimension*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
P
EqualEqualArgMax_2ArgMax_3*#
_output_shapes
:џџџџџџџџџ*
T0	
R
Cast_1CastEqual*#
_output_shapes
:џџџџџџџџџ*

SrcT0
*

DstT0
Q
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
_
accuracyMeanCast_1Const_1*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
[
ScalarSummary/tagsConst*
dtype0*
_output_shapes
: *
valueB BAccuracy
]
ScalarSummaryScalarSummaryScalarSummary/tagsaccuracy*
_output_shapes
: *
T0
Y
ScalarSummary_1/tagsConst*
dtype0*
_output_shapes
: *
valueB
 BLoss
]
ScalarSummary_1ScalarSummaryScalarSummary_1/tagscost*
_output_shapes
: *
T0
h
Reshape_5/shapeConst*
dtype0*
_output_shapes
:*%
valueB"џџџџИ   И      
r
	Reshape_5ReshapexReshape_5/shape*1
_output_shapes
:џџџџџџџџџИИ*
T0*
Tshape0
Y
ImageSummary/tagConst*
dtype0*
_output_shapes
: *
valueB BOriginal

ImageSummaryImageSummaryImageSummary/tag	Reshape_5*
_output_shapes
: *

max_images*
	bad_colorB:џ  џ*
T0
h
Reshape_6/shapeConst*
dtype0*
_output_shapes
:*%
valueB"џџџџИ   И      
r
	Reshape_6ReshapexReshape_6/shape*1
_output_shapes
:џџџџџџџџџИИ*
T0*
Tshape0
И
Conv2D_3Conv2D	Reshape_6wc1/read*
paddingSAME*
data_formatNHWC*
strides
*/
_output_shapes
:џџџџџџџџџ..*
T0*
use_cudnn_on_gpu(
y
	BiasAdd_3BiasAddConv2D_3bc1/read*/
_output_shapes
:џџџџџџџџџ..*
T0*
data_formatNHWC
S
Relu_4Relu	BiasAdd_3*/
_output_shapes
:џџџџџџџџџ..*
T0
L
conv1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
W
conv1AddRelu_4conv1/y*/
_output_shapes
:џџџџџџџџџ..*
T0
h
Reshape_7/shapeConst*
dtype0*
_output_shapes
:*%
valueB"џџџџ.   .      
t
	Reshape_7Reshapeconv1Reshape_7/shape*/
_output_shapes
:џџџџџџџџџ..*
T0*
Tshape0
Y
ImageSummary_1/tagConst*
dtype0*
_output_shapes
: *
valueB B1.Conv

ImageSummary_1ImageSummaryImageSummary_1/tag	Reshape_7*
_output_shapes
: *

max_images*
	bad_colorB:џ  џ*
T0
Д
Conv2D_4Conv2Dconv1wc2/read*
paddingSAME*
data_formatNHWC*
strides
*/
_output_shapes
:џџџџџџџџџ..*
T0*
use_cudnn_on_gpu(
y
	BiasAdd_4BiasAddConv2D_4bc2/read*/
_output_shapes
:џџџџџџџџџ..*
T0*
data_formatNHWC
S
Relu_5Relu	BiasAdd_4*/
_output_shapes
:џџџџџџџџџ..*
T0
L
conv2/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
W
conv2AddRelu_5conv2/y*/
_output_shapes
:џџџџџџџџџ..*
T0
h
Reshape_8/shapeConst*
dtype0*
_output_shapes
:*%
valueB"џџџџ.   .      
t
	Reshape_8Reshapeconv2Reshape_8/shape*/
_output_shapes
:џџџџџџџџџ..*
T0*
Tshape0
Y
ImageSummary_2/tagConst*
dtype0*
_output_shapes
: *
valueB B2.Conv

ImageSummary_2ImageSummaryImageSummary_2/tag	Reshape_8*
_output_shapes
: *

max_images*
	bad_colorB:џ  џ*
T0
Д
Conv2D_5Conv2Dconv2wc3/read*
paddingSAME*
data_formatNHWC*
strides
*/
_output_shapes
:џџџџџџџџџ.. *
T0*
use_cudnn_on_gpu(
y
	BiasAdd_5BiasAddConv2D_5bc3/read*/
_output_shapes
:џџџџџџџџџ.. *
T0*
data_formatNHWC
S
Relu_6Relu	BiasAdd_5*/
_output_shapes
:џџџџџџџџџ.. *
T0
L
conv3/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
W
conv3AddRelu_6conv3/y*/
_output_shapes
:џџџџџџџџџ.. *
T0
h
Reshape_9/shapeConst*
dtype0*
_output_shapes
:*%
valueB"џџџџ.   .      
t
	Reshape_9Reshapeconv3Reshape_9/shape*/
_output_shapes
:џџџџџџџџџ..*
T0*
Tshape0
Y
ImageSummary_3/tagConst*
dtype0*
_output_shapes
: *
valueB B3.Conv

ImageSummary_3ImageSummaryImageSummary_3/tag	Reshape_9*
_output_shapes
: *

max_images *
	bad_colorB:џ  џ*
T0
e
HistogramSummary/tagConst*
dtype0*
_output_shapes
: *!
valueB BHistogram 1.Conv
e
HistogramSummaryHistogramSummaryHistogramSummary/tagwc1/read*
_output_shapes
: *
T0
e
HistogramSummary_1/tagConst*
dtype0*
_output_shapes
: *
valueB BHistogram pred
e
HistogramSummary_1HistogramSummaryHistogramSummary_1/tagpred*
_output_shapes
: *
T0
Ю
MergeSummary/MergeSummaryMergeSummaryScalarSummaryScalarSummary_1ImageSummaryImageSummary_1ImageSummary_2ImageSummary_3HistogramSummaryHistogramSummary_1*
_output_shapes
: *
N
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel

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
Љ
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesb_outbc1bc2bc3bd1w_outwc1wc2wc3wd1*
dtypes
2

}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
_class
loc:@save/Const*
T0
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

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2

save/AssignAssignb_outsave/RestoreV2*
use_locking(*
_output_shapes
:*
validate_shape(*
_class

loc:@b_out*
T0
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

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_1Assignbc1save/RestoreV2_1*
use_locking(*
_output_shapes
:*
validate_shape(*
_class

loc:@bc1*
T0
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

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_2Assignbc2save/RestoreV2_2*
use_locking(*
_output_shapes
:*
validate_shape(*
_class

loc:@bc2*
T0
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

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_3Assignbc3save/RestoreV2_3*
use_locking(*
_output_shapes
: *
validate_shape(*
_class

loc:@bc3*
T0
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

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_4Assignbd1save/RestoreV2_4*
use_locking(*
_output_shapes	
:ш*
validate_shape(*
_class

loc:@bd1*
T0
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

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_5Assignw_outsave/RestoreV2_5*
use_locking(*
_output_shapes
:	ш*
validate_shape(*
_class

loc:@w_out*
T0
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

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
 
save/Assign_6Assignwc1save/RestoreV2_6*
use_locking(*&
_output_shapes
:*
validate_shape(*
_class

loc:@wc1*
T0
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

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
 
save/Assign_7Assignwc2save/RestoreV2_7*
use_locking(*&
_output_shapes
:*
validate_shape(*
_class

loc:@wc2*
T0
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

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
 
save/Assign_8Assignwc3save/RestoreV2_8*
use_locking(*&
_output_shapes
: *
validate_shape(*
_class

loc:@wc3*
T0
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

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_9Assignwd1save/RestoreV2_9*
use_locking(* 
_output_shapes
:
	ш*
validate_shape(*
_class

loc:@wd1*
T0
Ж
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9

initNoOp^wc1/Assign^wc2/Assign^wc3/Assign^wd1/Assign^w_out/Assign^bc1/Assign^bc2/Assign^bc3/Assign^bd1/Assign^b_out/Assign"ал5E     UБы	L4 єnзAJЈ
П))
9
Add
x"T
y"T
z"T"
Ttype:
2	

ApplyGradientDescent
var"T

alpha"T

delta"T
out"T"
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
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
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
Щ
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
я
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
ю
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

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

ImageSummary
tag
tensor"T
summary"

max_imagesint(0"
Ttype0:
2"'
	bad_colortensorB:џ  џ
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
О
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
ф
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
2	

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
2	
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
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
і
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

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
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*0.12.020.12.0-rc1-7-ga13284f-dirtyЛй
Y
xPlaceholder*
dtype0*-
_output_shapes
:џџџџџџџџџР*
shape: 
S
yPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape: 
R
	keep_probPlaceholder*
dtype0*
_output_shapes
:*
shape:
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
 *ЭЬЬ=
Љ
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
dtype0*
seed2*
seedБџх)*
T0*&
_output_shapes
:

random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*&
_output_shapes
:*
T0
l
random_normalAddrandom_normal/mulrandom_normal/mean*&
_output_shapes
:*
T0

wc1Variable*
dtype0*
shared_name *
shape:*
	container *&
_output_shapes
:


wc1/AssignAssignwc1random_normal*
use_locking(*&
_output_shapes
:*
validate_shape(*
_class

loc:@wc1*
T0
b
wc1/readIdentitywc1*&
_output_shapes
:*
_class

loc:@wc1*
T0
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
 *ЭЬЬ=
­
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
dtype0*
seed2*
seedБџх)*
T0*&
_output_shapes
:

random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*&
_output_shapes
:*
T0
r
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*&
_output_shapes
:*
T0

wc2Variable*
dtype0*
shared_name *
shape:*
	container *&
_output_shapes
:


wc2/AssignAssignwc2random_normal_1*
use_locking(*&
_output_shapes
:*
validate_shape(*
_class

loc:@wc2*
T0
b
wc2/readIdentitywc2*&
_output_shapes
:*
_class

loc:@wc2*
T0
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
 *ЭЬЬ=
­
$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*
dtype0*
seed2*
seedБџх)*
T0*&
_output_shapes
: 

random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*&
_output_shapes
: *
T0
r
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*&
_output_shapes
: *
T0

wc3Variable*
dtype0*
shared_name *
shape: *
	container *&
_output_shapes
: 


wc3/AssignAssignwc3random_normal_2*
use_locking(*&
_output_shapes
: *
validate_shape(*
_class

loc:@wc3*
T0
b
wc3/readIdentitywc3*&
_output_shapes
: *
_class

loc:@wc3*
T0
f
random_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"  h  
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
 *ЭЬЬ=
Ї
$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*
dtype0*
seed2*
seedБџх)*
T0* 
_output_shapes
:
	ш

random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev* 
_output_shapes
:
	ш*
T0
l
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean* 
_output_shapes
:
	ш*
T0
y
wd1Variable*
dtype0*
shared_name *
shape:
	ш*
	container * 
_output_shapes
:
	ш


wd1/AssignAssignwd1random_normal_3*
use_locking(* 
_output_shapes
:
	ш*
validate_shape(*
_class

loc:@wd1*
T0
\
wd1/readIdentitywd1* 
_output_shapes
:
	ш*
_class

loc:@wd1*
T0
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
 *ЭЬЬ=
І
$random_normal_4/RandomStandardNormalRandomStandardNormalrandom_normal_4/shape*
dtype0*
seed2*
seedБџх)*
T0*
_output_shapes
:	ш

random_normal_4/mulMul$random_normal_4/RandomStandardNormalrandom_normal_4/stddev*
_output_shapes
:	ш*
T0
k
random_normal_4Addrandom_normal_4/mulrandom_normal_4/mean*
_output_shapes
:	ш*
T0
y
w_outVariable*
dtype0*
shared_name *
shape:	ш*
	container *
_output_shapes
:	ш

w_out/AssignAssignw_outrandom_normal_4*
use_locking(*
_output_shapes
:	ш*
validate_shape(*
_class

loc:@w_out*
T0
a

w_out/readIdentityw_out*
_output_shapes
:	ш*
_class

loc:@w_out*
T0
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
 *  ?

$random_normal_5/RandomStandardNormalRandomStandardNormalrandom_normal_5/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:
}
random_normal_5/mulMul$random_normal_5/RandomStandardNormalrandom_normal_5/stddev*
_output_shapes
:*
T0
f
random_normal_5Addrandom_normal_5/mulrandom_normal_5/mean*
_output_shapes
:*
T0
m
bc1Variable*
dtype0*
shared_name *
shape:*
	container *
_output_shapes
:


bc1/AssignAssignbc1random_normal_5*
use_locking(*
_output_shapes
:*
validate_shape(*
_class

loc:@bc1*
T0
V
bc1/readIdentitybc1*
_output_shapes
:*
_class

loc:@bc1*
T0
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
 *  ?

$random_normal_6/RandomStandardNormalRandomStandardNormalrandom_normal_6/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:
}
random_normal_6/mulMul$random_normal_6/RandomStandardNormalrandom_normal_6/stddev*
_output_shapes
:*
T0
f
random_normal_6Addrandom_normal_6/mulrandom_normal_6/mean*
_output_shapes
:*
T0
m
bc2Variable*
dtype0*
shared_name *
shape:*
	container *
_output_shapes
:


bc2/AssignAssignbc2random_normal_6*
use_locking(*
_output_shapes
:*
validate_shape(*
_class

loc:@bc2*
T0
V
bc2/readIdentitybc2*
_output_shapes
:*
_class

loc:@bc2*
T0
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
 *  ?

$random_normal_7/RandomStandardNormalRandomStandardNormalrandom_normal_7/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
: 
}
random_normal_7/mulMul$random_normal_7/RandomStandardNormalrandom_normal_7/stddev*
_output_shapes
: *
T0
f
random_normal_7Addrandom_normal_7/mulrandom_normal_7/mean*
_output_shapes
: *
T0
m
bc3Variable*
dtype0*
shared_name *
shape: *
	container *
_output_shapes
: 


bc3/AssignAssignbc3random_normal_7*
use_locking(*
_output_shapes
: *
validate_shape(*
_class

loc:@bc3*
T0
V
bc3/readIdentitybc3*
_output_shapes
: *
_class

loc:@bc3*
T0
`
random_normal_8/shapeConst*
dtype0*
_output_shapes
:*
valueB:ш
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
 *  ?

$random_normal_8/RandomStandardNormalRandomStandardNormalrandom_normal_8/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes	
:ш
~
random_normal_8/mulMul$random_normal_8/RandomStandardNormalrandom_normal_8/stddev*
_output_shapes	
:ш*
T0
g
random_normal_8Addrandom_normal_8/mulrandom_normal_8/mean*
_output_shapes	
:ш*
T0
o
bd1Variable*
dtype0*
shared_name *
shape:ш*
	container *
_output_shapes	
:ш


bd1/AssignAssignbd1random_normal_8*
use_locking(*
_output_shapes	
:ш*
validate_shape(*
_class

loc:@bd1*
T0
W
bd1/readIdentitybd1*
_output_shapes	
:ш*
_class

loc:@bd1*
T0
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
 *  ?

$random_normal_9/RandomStandardNormalRandomStandardNormalrandom_normal_9/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:
}
random_normal_9/mulMul$random_normal_9/RandomStandardNormalrandom_normal_9/stddev*
_output_shapes
:*
T0
f
random_normal_9Addrandom_normal_9/mulrandom_normal_9/mean*
_output_shapes
:*
T0
o
b_outVariable*
dtype0*
shared_name *
shape:*
	container *
_output_shapes
:

b_out/AssignAssignb_outrandom_normal_9*
use_locking(*
_output_shapes
:*
validate_shape(*
_class

loc:@b_out*
T0
\

b_out/readIdentityb_out*
_output_shapes
:*
_class

loc:@b_out*
T0
f
Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"џџџџИ   И      
n
ReshapeReshapexReshape/shape*1
_output_shapes
:џџџџџџџџџИИ*
T0*
Tshape0
Д
Conv2DConv2DReshapewc1/read*
paddingSAME*
data_formatNHWC*
strides
*/
_output_shapes
:џџџџџџџџџ..*
T0*
use_cudnn_on_gpu(
u
BiasAddBiasAddConv2Dbc1/read*/
_output_shapes
:џџџџџџџџџ..*
data_formatNHWC*
T0
O
ReluReluBiasAdd*/
_output_shapes
:џџџџџџџџџ..*
T0
Є
MaxPoolMaxPoolRelu*
paddingSAME*
strides
*/
_output_shapes
:џџџџџџџџџ*
ksize
*
T0*
data_formatNHWC
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
§
strided_sliceStridedSlice	keep_probstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
new_axis_mask *
Index0*
ellipsis_mask *
_output_shapes
: *
shrink_axis_mask*
end_mask *
T0*

begin_mask 
T
dropout/ShapeShapeMaxPool*
out_type0*
T0*
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
 *  ?
Є
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
dtype0*
seed2 *

seed *
T0*/
_output_shapes
:џџџџџџџџџ
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0

dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*/
_output_shapes
:џџџџџџџџџ*
T0

dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*/
_output_shapes
:џџџџџџџџџ*
T0
s
dropout/addAddstrided_slicedropout/random_uniform*/
_output_shapes
:џџџџџџџџџ*
T0
]
dropout/FloorFloordropout/add*/
_output_shapes
:џџџџџџџџџ*
T0
d
dropout/DivDivMaxPoolstrided_slice*/
_output_shapes
:џџџџџџџџџ*
T0
h
dropout/mulMuldropout/Divdropout/Floor*/
_output_shapes
:џџџџџџџџџ*
T0
К
Conv2D_1Conv2Ddropout/mulwc2/read*
paddingSAME*
data_formatNHWC*
strides
*/
_output_shapes
:џџџџџџџџџ*
T0*
use_cudnn_on_gpu(
y
	BiasAdd_1BiasAddConv2D_1bc2/read*/
_output_shapes
:џџџџџџџџџ*
data_formatNHWC*
T0
S
Relu_1Relu	BiasAdd_1*/
_output_shapes
:џџџџџџџџџ*
T0
Ј
	MaxPool_1MaxPoolRelu_1*
paddingSAME*
strides
*/
_output_shapes
:џџџџџџџџџ*
ksize
*
T0*
data_formatNHWC
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

strided_slice_1StridedSlice	keep_probstrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
new_axis_mask *
Index0*
ellipsis_mask *
_output_shapes
: *
shrink_axis_mask*
end_mask *
T0*

begin_mask 
X
dropout_1/ShapeShape	MaxPool_1*
out_type0*
T0*
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
 *  ?
Ј
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape*
dtype0*
seed2 *

seed *
T0*/
_output_shapes
:џџџџџџџџџ

dropout_1/random_uniform/subSubdropout_1/random_uniform/maxdropout_1/random_uniform/min*
_output_shapes
: *
T0
Ѓ
dropout_1/random_uniform/mulMul&dropout_1/random_uniform/RandomUniformdropout_1/random_uniform/sub*/
_output_shapes
:џџџџџџџџџ*
T0

dropout_1/random_uniformAdddropout_1/random_uniform/muldropout_1/random_uniform/min*/
_output_shapes
:џџџџџџџџџ*
T0
y
dropout_1/addAddstrided_slice_1dropout_1/random_uniform*/
_output_shapes
:џџџџџџџџџ*
T0
a
dropout_1/FloorFloordropout_1/add*/
_output_shapes
:џџџџџџџџџ*
T0
j
dropout_1/DivDiv	MaxPool_1strided_slice_1*/
_output_shapes
:џџџџџџџџџ*
T0
n
dropout_1/mulMuldropout_1/Divdropout_1/Floor*/
_output_shapes
:џџџџџџџџџ*
T0
М
Conv2D_2Conv2Ddropout_1/mulwc3/read*
paddingSAME*
data_formatNHWC*
strides
*/
_output_shapes
:џџџџџџџџџ *
T0*
use_cudnn_on_gpu(
y
	BiasAdd_2BiasAddConv2D_2bc3/read*/
_output_shapes
:џџџџџџџџџ *
data_formatNHWC*
T0
S
Relu_2Relu	BiasAdd_2*/
_output_shapes
:џџџџџџџџџ *
T0
Ј
	MaxPool_2MaxPoolRelu_2*
paddingSAME*
strides
*/
_output_shapes
:џџџџџџџџџ *
ksize
*
T0*
data_formatNHWC
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

strided_slice_2StridedSlice	keep_probstrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
new_axis_mask *
Index0*
ellipsis_mask *
_output_shapes
: *
shrink_axis_mask*
end_mask *
T0*

begin_mask 
X
dropout_2/ShapeShape	MaxPool_2*
out_type0*
T0*
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
 *  ?
Ј
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape*
dtype0*
seed2 *

seed *
T0*/
_output_shapes
:џџџџџџџџџ 

dropout_2/random_uniform/subSubdropout_2/random_uniform/maxdropout_2/random_uniform/min*
_output_shapes
: *
T0
Ѓ
dropout_2/random_uniform/mulMul&dropout_2/random_uniform/RandomUniformdropout_2/random_uniform/sub*/
_output_shapes
:џџџџџџџџџ *
T0

dropout_2/random_uniformAdddropout_2/random_uniform/muldropout_2/random_uniform/min*/
_output_shapes
:џџџџџџџџџ *
T0
y
dropout_2/addAddstrided_slice_2dropout_2/random_uniform*/
_output_shapes
:џџџџџџџџџ *
T0
a
dropout_2/FloorFloordropout_2/add*/
_output_shapes
:џџџџџџџџџ *
T0
j
dropout_2/DivDiv	MaxPool_2strided_slice_2*/
_output_shapes
:џџџџџџџџџ *
T0
n
dropout_2/mulMuldropout_2/Divdropout_2/Floor*/
_output_shapes
:џџџџџџџџџ *
T0
`
Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ  
u
	Reshape_1Reshapedropout_2/mulReshape_1/shape*(
_output_shapes
:џџџџџџџџџ	*
T0*
Tshape0
~
MatMulMatMul	Reshape_1wd1/read*(
_output_shapes
:џџџџџџџџџш*
transpose_a( *
transpose_b( *
T0
O
AddAddMatMulbd1/read*(
_output_shapes
:џџџџџџџџџш*
T0
F
Relu_3ReluAdd*(
_output_shapes
:џџџџџџџџџш*
T0
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

strided_slice_3StridedSlice	keep_probstrided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_2*
new_axis_mask *
Index0*
ellipsis_mask *
_output_shapes
: *
shrink_axis_mask*
end_mask *
T0*

begin_mask 
U
dropout_3/ShapeShapeRelu_3*
out_type0*
T0*
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
 *  ?
Ё
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:џџџџџџџџџш

dropout_3/random_uniform/subSubdropout_3/random_uniform/maxdropout_3/random_uniform/min*
_output_shapes
: *
T0

dropout_3/random_uniform/mulMul&dropout_3/random_uniform/RandomUniformdropout_3/random_uniform/sub*(
_output_shapes
:џџџџџџџџџш*
T0

dropout_3/random_uniformAdddropout_3/random_uniform/muldropout_3/random_uniform/min*(
_output_shapes
:џџџџџџџџџш*
T0
r
dropout_3/addAddstrided_slice_3dropout_3/random_uniform*(
_output_shapes
:џџџџџџџџџш*
T0
Z
dropout_3/FloorFloordropout_3/add*(
_output_shapes
:џџџџџџџџџш*
T0
`
dropout_3/DivDivRelu_3strided_slice_3*(
_output_shapes
:џџџџџџџџџш*
T0
g
dropout_3/mulMuldropout_3/Divdropout_3/Floor*(
_output_shapes
:џџџџџџџџџш*
T0

MatMul_1MatMuldropout_3/mul
w_out/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
T
Add_1AddMatMul_1
b_out/read*'
_output_shapes
:џџџџџџџџџ*
T0
K
pred/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
L
predAddAdd_1pred/y*'
_output_shapes
:џџџџџџџџџ*
T0
F
RankConst*
dtype0*
_output_shapes
: *
value	B :
I
ShapeShapepred*
out_type0*
T0*
_output_shapes
:
H
Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
K
Shape_1Shapepred*
out_type0*
T0*
_output_shapes
:
G
Sub/yConst*
dtype0*
_output_shapes
: *
value	B :
:
SubSubRank_1Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*
_output_shapes
:*
N*

axis *
T0
T

Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
b
SliceSliceShape_1Slice/begin
Slice/size*
_output_shapes
:*
Index0*
T0
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
џџџџџџџџџ
i
concatConcatconcat/concat_dimconcat/values_0Slice*
_output_shapes
:*
N*
T0
k
	Reshape_2Reshapepredconcat*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*
Tshape0
H
Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
H
Shape_2Shapey*
out_type0*
T0*
_output_shapes
:
I
Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
>
Sub_1SubRank_2Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*
_output_shapes
:*
N*

axis *
T0
V
Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
_output_shapes
:*
Index0*
T0
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
џџџџџџџџџ
q
concat_1Concatconcat_1/concat_dimconcat_1/values_0Slice_1*
_output_shapes
:*
N*
T0
j
	Reshape_3Reshapeyconcat_1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*
Tshape0

SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ*
T0
I
Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB: 
U
Slice_2/sizePackSub_2*
_output_shapes
:*
N*

axis *
T0
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*#
_output_shapes
:џџџџџџџџџ*
Index0*
T0
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
O
ConstConst*
dtype0*
_output_shapes
:*
valueB: 
\
costMean	Reshape_4Const*
_output_shapes
: *
T0*
	keep_dims( *

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
 *  ?
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
k
!gradients/cost_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:

gradients/cost_grad/ReshapeReshapegradients/Fill!gradients/cost_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
b
gradients/cost_grad/ShapeShape	Reshape_4*
out_type0*
T0*
_output_shapes
:

gradients/cost_grad/TileTilegradients/cost_grad/Reshapegradients/cost_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:џџџџџџџџџ
d
gradients/cost_grad/Shape_1Shape	Reshape_4*
out_type0*
T0*
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

gradients/cost_grad/ProdProdgradients/cost_grad/Shape_1gradients/cost_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
e
gradients/cost_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 

gradients/cost_grad/Prod_1Prodgradients/cost_grad/Shape_2gradients/cost_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
_
gradients/cost_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :

gradients/cost_grad/MaximumMaximumgradients/cost_grad/Prod_1gradients/cost_grad/Maximum/y*
_output_shapes
: *
T0
{
gradients/cost_grad/floordivDivgradients/cost_grad/Prodgradients/cost_grad/Maximum*
_output_shapes
: *
T0
n
gradients/cost_grad/CastCastgradients/cost_grad/floordiv*
_output_shapes
: *

SrcT0*

DstT0

gradients/cost_grad/truedivDivgradients/cost_grad/Tilegradients/cost_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
out_type0*
T0*
_output_shapes
:
Є
 gradients/Reshape_4_grad/ReshapeReshapegradients/cost_grad/truedivgradients/Reshape_4_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0

;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
т
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
Ь
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
b
gradients/Reshape_2_grad/ShapeShapepred*
out_type0*
T0*
_output_shapes
:
Н
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
^
gradients/pred_grad/ShapeShapeAdd_1*
out_type0*
T0*
_output_shapes
:
^
gradients/pred_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
З
)gradients/pred_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pred_grad/Shapegradients/pred_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/pred_grad/SumSum gradients/Reshape_2_grad/Reshape)gradients/pred_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/pred_grad/ReshapeReshapegradients/pred_grad/Sumgradients/pred_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Џ
gradients/pred_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape+gradients/pred_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/pred_grad/Reshape_1Reshapegradients/pred_grad/Sum_1gradients/pred_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
j
$gradients/pred_grad/tuple/group_depsNoOp^gradients/pred_grad/Reshape^gradients/pred_grad/Reshape_1
о
,gradients/pred_grad/tuple/control_dependencyIdentitygradients/pred_grad/Reshape%^gradients/pred_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*.
_class$
" loc:@gradients/pred_grad/Reshape*
T0
г
.gradients/pred_grad/tuple/control_dependency_1Identitygradients/pred_grad/Reshape_1%^gradients/pred_grad/tuple/group_deps*
_output_shapes
: *0
_class&
$"loc:@gradients/pred_grad/Reshape_1*
T0
b
gradients/Add_1_grad/ShapeShapeMatMul_1*
out_type0*
T0*
_output_shapes
:
f
gradients/Add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
К
*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Й
gradients/Add_1_grad/SumSum,gradients/pred_grad/tuple/control_dependency*gradients/Add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Н
gradients/Add_1_grad/Sum_1Sum,gradients/pred_grad/tuple/control_dependency,gradients/Add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
т
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*/
_class%
#!loc:@gradients/Add_1_grad/Reshape*
T0
л
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*
_output_shapes
:*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1*
T0
М
gradients/MatMul_1_grad/MatMulMatMul-gradients/Add_1_grad/tuple/control_dependency
w_out/read*(
_output_shapes
:џџџџџџџџџш*
transpose_a( *
transpose_b(*
T0
И
 gradients/MatMul_1_grad/MatMul_1MatMuldropout_3/mul-gradients/Add_1_grad/tuple/control_dependency*
_output_shapes
:	ш*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
э
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџш*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0
ъ
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
_output_shapes
:	ш*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0
o
"gradients/dropout_3/mul_grad/ShapeShapedropout_3/Div*
out_type0*
T0*
_output_shapes
:
s
$gradients/dropout_3/mul_grad/Shape_1Shapedropout_3/Floor*
out_type0*
T0*
_output_shapes
:
в
2gradients/dropout_3/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_3/mul_grad/Shape$gradients/dropout_3/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

 gradients/dropout_3/mul_grad/mulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout_3/Floor*(
_output_shapes
:џџџџџџџџџш*
T0
Н
 gradients/dropout_3/mul_grad/SumSum gradients/dropout_3/mul_grad/mul2gradients/dropout_3/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ж
$gradients/dropout_3/mul_grad/ReshapeReshape gradients/dropout_3/mul_grad/Sum"gradients/dropout_3/mul_grad/Shape*(
_output_shapes
:џџџџџџџџџш*
T0*
Tshape0

"gradients/dropout_3/mul_grad/mul_1Muldropout_3/Div0gradients/MatMul_1_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџш*
T0
У
"gradients/dropout_3/mul_grad/Sum_1Sum"gradients/dropout_3/mul_grad/mul_14gradients/dropout_3/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
М
&gradients/dropout_3/mul_grad/Reshape_1Reshape"gradients/dropout_3/mul_grad/Sum_1$gradients/dropout_3/mul_grad/Shape_1*(
_output_shapes
:џџџџџџџџџш*
T0*
Tshape0

-gradients/dropout_3/mul_grad/tuple/group_depsNoOp%^gradients/dropout_3/mul_grad/Reshape'^gradients/dropout_3/mul_grad/Reshape_1

5gradients/dropout_3/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_3/mul_grad/Reshape.^gradients/dropout_3/mul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџш*7
_class-
+)loc:@gradients/dropout_3/mul_grad/Reshape*
T0

7gradients/dropout_3/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_3/mul_grad/Reshape_1.^gradients/dropout_3/mul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџш*9
_class/
-+loc:@gradients/dropout_3/mul_grad/Reshape_1*
T0
h
"gradients/dropout_3/Div_grad/ShapeShapeRelu_3*
out_type0*
T0*
_output_shapes
:
g
$gradients/dropout_3/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
в
2gradients/dropout_3/Div_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_3/Div_grad/Shape$gradients/dropout_3/Div_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
І
$gradients/dropout_3/Div_grad/truedivDiv5gradients/dropout_3/mul_grad/tuple/control_dependencystrided_slice_3*(
_output_shapes
:џџџџџџџџџш*
T0
С
 gradients/dropout_3/Div_grad/SumSum$gradients/dropout_3/Div_grad/truediv2gradients/dropout_3/Div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ж
$gradients/dropout_3/Div_grad/ReshapeReshape gradients/dropout_3/Div_grad/Sum"gradients/dropout_3/Div_grad/Shape*(
_output_shapes
:џџџџџџџџџш*
T0*
Tshape0
b
 gradients/dropout_3/Div_grad/NegNegRelu_3*(
_output_shapes
:џџџџџџџџџш*
T0
_
#gradients/dropout_3/Div_grad/SquareSquarestrided_slice_3*
_output_shapes
: *
T0
Ї
&gradients/dropout_3/Div_grad/truediv_1Div gradients/dropout_3/Div_grad/Neg#gradients/dropout_3/Div_grad/Square*(
_output_shapes
:џџџџџџџџџш*
T0
Й
 gradients/dropout_3/Div_grad/mulMul5gradients/dropout_3/mul_grad/tuple/control_dependency&gradients/dropout_3/Div_grad/truediv_1*(
_output_shapes
:џџџџџџџџџш*
T0
С
"gradients/dropout_3/Div_grad/Sum_1Sum gradients/dropout_3/Div_grad/mul4gradients/dropout_3/Div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Њ
&gradients/dropout_3/Div_grad/Reshape_1Reshape"gradients/dropout_3/Div_grad/Sum_1$gradients/dropout_3/Div_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

-gradients/dropout_3/Div_grad/tuple/group_depsNoOp%^gradients/dropout_3/Div_grad/Reshape'^gradients/dropout_3/Div_grad/Reshape_1

5gradients/dropout_3/Div_grad/tuple/control_dependencyIdentity$gradients/dropout_3/Div_grad/Reshape.^gradients/dropout_3/Div_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџш*7
_class-
+)loc:@gradients/dropout_3/Div_grad/Reshape*
T0
ї
7gradients/dropout_3/Div_grad/tuple/control_dependency_1Identity&gradients/dropout_3/Div_grad/Reshape_1.^gradients/dropout_3/Div_grad/tuple/group_deps*
_output_shapes
: *9
_class/
-+loc:@gradients/dropout_3/Div_grad/Reshape_1*
T0

gradients/Relu_3_grad/ReluGradReluGrad5gradients/dropout_3/Div_grad/tuple/control_dependencyRelu_3*(
_output_shapes
:џџџџџџџџџш*
T0
^
gradients/Add_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
e
gradients/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:ш
Д
(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ї
gradients/Add_grad/SumSumgradients/Relu_3_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*(
_output_shapes
:џџџџџџџџџш*
T0*
Tshape0
Ћ
gradients/Add_grad/Sum_1Sumgradients/Relu_3_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
_output_shapes	
:ш*
T0*
Tshape0
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
л
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџш*-
_class#
!loc:@gradients/Add_grad/Reshape*
T0
д
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*
_output_shapes	
:ш*/
_class%
#!loc:@gradients/Add_grad/Reshape_1*
T0
Ж
gradients/MatMul_grad/MatMulMatMul+gradients/Add_grad/tuple/control_dependencywd1/read*(
_output_shapes
:џџџџџџџџџ	*
transpose_a( *
transpose_b(*
T0
Б
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1+gradients/Add_grad/tuple/control_dependency* 
_output_shapes
:
	ш*
transpose_a(*
transpose_b( *
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
х
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ	*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0
у
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps* 
_output_shapes
:
	ш*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
k
gradients/Reshape_1_grad/ShapeShapedropout_2/mul*
out_type0*
T0*
_output_shapes
:
У
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*/
_output_shapes
:џџџџџџџџџ *
T0*
Tshape0
o
"gradients/dropout_2/mul_grad/ShapeShapedropout_2/Div*
out_type0*
T0*
_output_shapes
:
s
$gradients/dropout_2/mul_grad/Shape_1Shapedropout_2/Floor*
out_type0*
T0*
_output_shapes
:
в
2gradients/dropout_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_2/mul_grad/Shape$gradients/dropout_2/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

 gradients/dropout_2/mul_grad/mulMul gradients/Reshape_1_grad/Reshapedropout_2/Floor*/
_output_shapes
:џџџџџџџџџ *
T0
Н
 gradients/dropout_2/mul_grad/SumSum gradients/dropout_2/mul_grad/mul2gradients/dropout_2/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Н
$gradients/dropout_2/mul_grad/ReshapeReshape gradients/dropout_2/mul_grad/Sum"gradients/dropout_2/mul_grad/Shape*/
_output_shapes
:џџџџџџџџџ *
T0*
Tshape0

"gradients/dropout_2/mul_grad/mul_1Muldropout_2/Div gradients/Reshape_1_grad/Reshape*/
_output_shapes
:џџџџџџџџџ *
T0
У
"gradients/dropout_2/mul_grad/Sum_1Sum"gradients/dropout_2/mul_grad/mul_14gradients/dropout_2/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
&gradients/dropout_2/mul_grad/Reshape_1Reshape"gradients/dropout_2/mul_grad/Sum_1$gradients/dropout_2/mul_grad/Shape_1*/
_output_shapes
:џџџџџџџџџ *
T0*
Tshape0

-gradients/dropout_2/mul_grad/tuple/group_depsNoOp%^gradients/dropout_2/mul_grad/Reshape'^gradients/dropout_2/mul_grad/Reshape_1

5gradients/dropout_2/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_2/mul_grad/Reshape.^gradients/dropout_2/mul_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ *7
_class-
+)loc:@gradients/dropout_2/mul_grad/Reshape*
T0

7gradients/dropout_2/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_2/mul_grad/Reshape_1.^gradients/dropout_2/mul_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ *9
_class/
-+loc:@gradients/dropout_2/mul_grad/Reshape_1*
T0
k
"gradients/dropout_2/Div_grad/ShapeShape	MaxPool_2*
out_type0*
T0*
_output_shapes
:
g
$gradients/dropout_2/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
в
2gradients/dropout_2/Div_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_2/Div_grad/Shape$gradients/dropout_2/Div_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
­
$gradients/dropout_2/Div_grad/truedivDiv5gradients/dropout_2/mul_grad/tuple/control_dependencystrided_slice_2*/
_output_shapes
:џџџџџџџџџ *
T0
С
 gradients/dropout_2/Div_grad/SumSum$gradients/dropout_2/Div_grad/truediv2gradients/dropout_2/Div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Н
$gradients/dropout_2/Div_grad/ReshapeReshape gradients/dropout_2/Div_grad/Sum"gradients/dropout_2/Div_grad/Shape*/
_output_shapes
:џџџџџџџџџ *
T0*
Tshape0
l
 gradients/dropout_2/Div_grad/NegNeg	MaxPool_2*/
_output_shapes
:џџџџџџџџџ *
T0
_
#gradients/dropout_2/Div_grad/SquareSquarestrided_slice_2*
_output_shapes
: *
T0
Ў
&gradients/dropout_2/Div_grad/truediv_1Div gradients/dropout_2/Div_grad/Neg#gradients/dropout_2/Div_grad/Square*/
_output_shapes
:џџџџџџџџџ *
T0
Р
 gradients/dropout_2/Div_grad/mulMul5gradients/dropout_2/mul_grad/tuple/control_dependency&gradients/dropout_2/Div_grad/truediv_1*/
_output_shapes
:џџџџџџџџџ *
T0
С
"gradients/dropout_2/Div_grad/Sum_1Sum gradients/dropout_2/Div_grad/mul4gradients/dropout_2/Div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Њ
&gradients/dropout_2/Div_grad/Reshape_1Reshape"gradients/dropout_2/Div_grad/Sum_1$gradients/dropout_2/Div_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

-gradients/dropout_2/Div_grad/tuple/group_depsNoOp%^gradients/dropout_2/Div_grad/Reshape'^gradients/dropout_2/Div_grad/Reshape_1

5gradients/dropout_2/Div_grad/tuple/control_dependencyIdentity$gradients/dropout_2/Div_grad/Reshape.^gradients/dropout_2/Div_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ *7
_class-
+)loc:@gradients/dropout_2/Div_grad/Reshape*
T0
ї
7gradients/dropout_2/Div_grad/tuple/control_dependency_1Identity&gradients/dropout_2/Div_grad/Reshape_1.^gradients/dropout_2/Div_grad/tuple/group_deps*
_output_shapes
: *9
_class/
-+loc:@gradients/dropout_2/Div_grad/Reshape_1*
T0

$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_25gradients/dropout_2/Div_grad/tuple/control_dependency*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ *
data_formatNHWC*
ksize
*
T0

gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*/
_output_shapes
:џџџџџџџџџ *
T0

$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
_output_shapes
: *
data_formatNHWC*
T0
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
і
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ *1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*
T0
я
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
_output_shapes
: *7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad*
T0
j
gradients/Conv2D_2_grad/ShapeShapedropout_1/mul*
out_type0*
T0*
_output_shapes
:
Ъ
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/Shapewc3/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
strides
*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
use_cudnn_on_gpu(
x
gradients/Conv2D_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"             
Џ
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterdropout_1/mulgradients/Conv2D_2_grad/Shape_11gradients/BiasAdd_2_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
strides
*&
_output_shapes
: *
T0*
use_cudnn_on_gpu(

(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter

0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*
T0

2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*&
_output_shapes
: *?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*
T0
o
"gradients/dropout_1/mul_grad/ShapeShapedropout_1/Div*
out_type0*
T0*
_output_shapes
:
s
$gradients/dropout_1/mul_grad/Shape_1Shapedropout_1/Floor*
out_type0*
T0*
_output_shapes
:
в
2gradients/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_1/mul_grad/Shape$gradients/dropout_1/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Є
 gradients/dropout_1/mul_grad/mulMul0gradients/Conv2D_2_grad/tuple/control_dependencydropout_1/Floor*/
_output_shapes
:џџџџџџџџџ*
T0
Н
 gradients/dropout_1/mul_grad/SumSum gradients/dropout_1/mul_grad/mul2gradients/dropout_1/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Н
$gradients/dropout_1/mul_grad/ReshapeReshape gradients/dropout_1/mul_grad/Sum"gradients/dropout_1/mul_grad/Shape*/
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Є
"gradients/dropout_1/mul_grad/mul_1Muldropout_1/Div0gradients/Conv2D_2_grad/tuple/control_dependency*/
_output_shapes
:џџџџџџџџџ*
T0
У
"gradients/dropout_1/mul_grad/Sum_1Sum"gradients/dropout_1/mul_grad/mul_14gradients/dropout_1/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
&gradients/dropout_1/mul_grad/Reshape_1Reshape"gradients/dropout_1/mul_grad/Sum_1$gradients/dropout_1/mul_grad/Shape_1*/
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

-gradients/dropout_1/mul_grad/tuple/group_depsNoOp%^gradients/dropout_1/mul_grad/Reshape'^gradients/dropout_1/mul_grad/Reshape_1

5gradients/dropout_1/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_1/mul_grad/Reshape.^gradients/dropout_1/mul_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*7
_class-
+)loc:@gradients/dropout_1/mul_grad/Reshape*
T0

7gradients/dropout_1/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_1/mul_grad/Reshape_1.^gradients/dropout_1/mul_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*9
_class/
-+loc:@gradients/dropout_1/mul_grad/Reshape_1*
T0
k
"gradients/dropout_1/Div_grad/ShapeShape	MaxPool_1*
out_type0*
T0*
_output_shapes
:
g
$gradients/dropout_1/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
в
2gradients/dropout_1/Div_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_1/Div_grad/Shape$gradients/dropout_1/Div_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
­
$gradients/dropout_1/Div_grad/truedivDiv5gradients/dropout_1/mul_grad/tuple/control_dependencystrided_slice_1*/
_output_shapes
:џџџџџџџџџ*
T0
С
 gradients/dropout_1/Div_grad/SumSum$gradients/dropout_1/Div_grad/truediv2gradients/dropout_1/Div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Н
$gradients/dropout_1/Div_grad/ReshapeReshape gradients/dropout_1/Div_grad/Sum"gradients/dropout_1/Div_grad/Shape*/
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
l
 gradients/dropout_1/Div_grad/NegNeg	MaxPool_1*/
_output_shapes
:џџџџџџџџџ*
T0
_
#gradients/dropout_1/Div_grad/SquareSquarestrided_slice_1*
_output_shapes
: *
T0
Ў
&gradients/dropout_1/Div_grad/truediv_1Div gradients/dropout_1/Div_grad/Neg#gradients/dropout_1/Div_grad/Square*/
_output_shapes
:џџџџџџџџџ*
T0
Р
 gradients/dropout_1/Div_grad/mulMul5gradients/dropout_1/mul_grad/tuple/control_dependency&gradients/dropout_1/Div_grad/truediv_1*/
_output_shapes
:џџџџџџџџџ*
T0
С
"gradients/dropout_1/Div_grad/Sum_1Sum gradients/dropout_1/Div_grad/mul4gradients/dropout_1/Div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Њ
&gradients/dropout_1/Div_grad/Reshape_1Reshape"gradients/dropout_1/Div_grad/Sum_1$gradients/dropout_1/Div_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

-gradients/dropout_1/Div_grad/tuple/group_depsNoOp%^gradients/dropout_1/Div_grad/Reshape'^gradients/dropout_1/Div_grad/Reshape_1

5gradients/dropout_1/Div_grad/tuple/control_dependencyIdentity$gradients/dropout_1/Div_grad/Reshape.^gradients/dropout_1/Div_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*7
_class-
+)loc:@gradients/dropout_1/Div_grad/Reshape*
T0
ї
7gradients/dropout_1/Div_grad/tuple/control_dependency_1Identity&gradients/dropout_1/Div_grad/Reshape_1.^gradients/dropout_1/Div_grad/tuple/group_deps*
_output_shapes
: *9
_class/
-+loc:@gradients/dropout_1/Div_grad/Reshape_1*
T0

$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_15gradients/dropout_1/Div_grad/tuple/control_dependency*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ*
data_formatNHWC*
ksize
*
T0

gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*/
_output_shapes
:џџџџџџџџџ*
T0

$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
_output_shapes
:*
data_formatNHWC*
T0
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
і
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*
T0
я
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
_output_shapes
:*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
T0
h
gradients/Conv2D_1_grad/ShapeShapedropout/mul*
out_type0*
T0*
_output_shapes
:
Ъ
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/Shapewc2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
strides
*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
use_cudnn_on_gpu(
x
gradients/Conv2D_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            
­
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterdropout/mulgradients/Conv2D_1_grad/Shape_11gradients/BiasAdd_1_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
strides
*&
_output_shapes
:*
T0*
use_cudnn_on_gpu(

(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter

0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*
T0

2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*&
_output_shapes
:*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0
k
 gradients/dropout/mul_grad/ShapeShapedropout/Div*
out_type0*
T0*
_output_shapes
:
o
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
out_type0*
T0*
_output_shapes
:
Ь
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
 
gradients/dropout/mul_grad/mulMul0gradients/Conv2D_1_grad/tuple/control_dependencydropout/Floor*/
_output_shapes
:џџџџџџџџџ*
T0
З
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
З
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*/
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
 
 gradients/dropout/mul_grad/mul_1Muldropout/Div0gradients/Conv2D_1_grad/tuple/control_dependency*/
_output_shapes
:џџџџџџџџџ*
T0
Н
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Н
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*/
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1

3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*
T0

5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1*
T0
g
 gradients/dropout/Div_grad/ShapeShapeMaxPool*
out_type0*
T0*
_output_shapes
:
e
"gradients/dropout/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
Ь
0gradients/dropout/Div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/Div_grad/Shape"gradients/dropout/Div_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ї
"gradients/dropout/Div_grad/truedivDiv3gradients/dropout/mul_grad/tuple/control_dependencystrided_slice*/
_output_shapes
:џџџџџџџџџ*
T0
Л
gradients/dropout/Div_grad/SumSum"gradients/dropout/Div_grad/truediv0gradients/dropout/Div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
З
"gradients/dropout/Div_grad/ReshapeReshapegradients/dropout/Div_grad/Sum gradients/dropout/Div_grad/Shape*/
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
h
gradients/dropout/Div_grad/NegNegMaxPool*/
_output_shapes
:џџџџџџџџџ*
T0
[
!gradients/dropout/Div_grad/SquareSquarestrided_slice*
_output_shapes
: *
T0
Ј
$gradients/dropout/Div_grad/truediv_1Divgradients/dropout/Div_grad/Neg!gradients/dropout/Div_grad/Square*/
_output_shapes
:џџџџџџџџџ*
T0
К
gradients/dropout/Div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/Div_grad/truediv_1*/
_output_shapes
:џџџџџџџџџ*
T0
Л
 gradients/dropout/Div_grad/Sum_1Sumgradients/dropout/Div_grad/mul2gradients/dropout/Div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Є
$gradients/dropout/Div_grad/Reshape_1Reshape gradients/dropout/Div_grad/Sum_1"gradients/dropout/Div_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

+gradients/dropout/Div_grad/tuple/group_depsNoOp#^gradients/dropout/Div_grad/Reshape%^gradients/dropout/Div_grad/Reshape_1

3gradients/dropout/Div_grad/tuple/control_dependencyIdentity"gradients/dropout/Div_grad/Reshape,^gradients/dropout/Div_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*5
_class+
)'loc:@gradients/dropout/Div_grad/Reshape*
T0
я
5gradients/dropout/Div_grad/tuple/control_dependency_1Identity$gradients/dropout/Div_grad/Reshape_1,^gradients/dropout/Div_grad/tuple/group_deps*
_output_shapes
: *7
_class-
+)loc:@gradients/dropout/Div_grad/Reshape_1*
T0

"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool3gradients/dropout/Div_grad/tuple/control_dependency*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ..*
data_formatNHWC*
ksize
*
T0

gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*/
_output_shapes
:џџџџџџџџџ..*
T0

"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
_output_shapes
:*
data_formatNHWC*
T0
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
ю
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ..*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*
T0
ч
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
T0
b
gradients/Conv2D_grad/ShapeShapeReshape*
out_type0*
T0*
_output_shapes
:
Ф
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/Shapewc1/read/gradients/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
strides
*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
use_cudnn_on_gpu(
v
gradients/Conv2D_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            
Ѓ
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/Shape_1/gradients/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
strides
*&
_output_shapes
:*
T0*
use_cudnn_on_gpu(

&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter

.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*1
_output_shapes
:џџџџџџџџџИИ*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*
T0

0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*&
_output_shapes
:*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0
b
GradientDescent/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *ьQ8=
љ
/GradientDescent/update_wc1/ApplyGradientDescentApplyGradientDescentwc1GradientDescent/learning_rate0gradients/Conv2D_grad/tuple/control_dependency_1*
use_locking( *&
_output_shapes
:*
_class

loc:@wc1*
T0
ћ
/GradientDescent/update_wc2/ApplyGradientDescentApplyGradientDescentwc2GradientDescent/learning_rate2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_locking( *&
_output_shapes
:*
_class

loc:@wc2*
T0
ћ
/GradientDescent/update_wc3/ApplyGradientDescentApplyGradientDescentwc3GradientDescent/learning_rate2gradients/Conv2D_2_grad/tuple/control_dependency_1*
use_locking( *&
_output_shapes
: *
_class

loc:@wc3*
T0
ѓ
/GradientDescent/update_wd1/ApplyGradientDescentApplyGradientDescentwd1GradientDescent/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( * 
_output_shapes
:
	ш*
_class

loc:@wd1*
T0
њ
1GradientDescent/update_w_out/ApplyGradientDescentApplyGradientDescentw_outGradientDescent/learning_rate2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
_output_shapes
:	ш*
_class

loc:@w_out*
T0
ю
/GradientDescent/update_bc1/ApplyGradientDescentApplyGradientDescentbc1GradientDescent/learning_rate1gradients/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
_output_shapes
:*
_class

loc:@bc1*
T0
№
/GradientDescent/update_bc2/ApplyGradientDescentApplyGradientDescentbc2GradientDescent/learning_rate3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
use_locking( *
_output_shapes
:*
_class

loc:@bc2*
T0
№
/GradientDescent/update_bc3/ApplyGradientDescentApplyGradientDescentbc3GradientDescent/learning_rate3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
use_locking( *
_output_shapes
: *
_class

loc:@bc3*
T0
ы
/GradientDescent/update_bd1/ApplyGradientDescentApplyGradientDescentbd1GradientDescent/learning_rate-gradients/Add_grad/tuple/control_dependency_1*
use_locking( *
_output_shapes	
:ш*
_class

loc:@bd1*
T0
ђ
1GradientDescent/update_b_out/ApplyGradientDescentApplyGradientDescentb_outGradientDescent/learning_rate/gradients/Add_1_grad/tuple/control_dependency_1*
use_locking( *
_output_shapes
:*
_class

loc:@b_out*
T0

GradientDescentNoOp0^GradientDescent/update_wc1/ApplyGradientDescent0^GradientDescent/update_wc2/ApplyGradientDescent0^GradientDescent/update_wc3/ApplyGradientDescent0^GradientDescent/update_wd1/ApplyGradientDescent2^GradientDescent/update_w_out/ApplyGradientDescent0^GradientDescent/update_bc1/ApplyGradientDescent0^GradientDescent/update_bc2/ApplyGradientDescent0^GradientDescent/update_bc3/ApplyGradientDescent0^GradientDescent/update_bd1/ApplyGradientDescent2^GradientDescent/update_b_out/ApplyGradientDescent
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
b
ArgMaxArgMaxpredArgMax/dimension*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
T
ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
c
ArgMax_1ArgMaxyArgMax_1/dimension*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
T
ArgMax_2/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
f
ArgMax_2ArgMaxpredArgMax_2/dimension*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
T
ArgMax_3/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
c
ArgMax_3ArgMaxyArgMax_3/dimension*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
P
EqualEqualArgMax_2ArgMax_3*#
_output_shapes
:џџџџџџџџџ*
T0	
R
Cast_1CastEqual*#
_output_shapes
:џџџџџџџџџ*

SrcT0
*

DstT0
Q
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
_
accuracyMeanCast_1Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
[
ScalarSummary/tagsConst*
dtype0*
_output_shapes
: *
valueB BAccuracy
]
ScalarSummaryScalarSummaryScalarSummary/tagsaccuracy*
_output_shapes
: *
T0
Y
ScalarSummary_1/tagsConst*
dtype0*
_output_shapes
: *
valueB
 BLoss
]
ScalarSummary_1ScalarSummaryScalarSummary_1/tagscost*
_output_shapes
: *
T0
h
Reshape_5/shapeConst*
dtype0*
_output_shapes
:*%
valueB"џџџџИ   И      
r
	Reshape_5ReshapexReshape_5/shape*1
_output_shapes
:џџџџџџџџџИИ*
T0*
Tshape0
Y
ImageSummary/tagConst*
dtype0*
_output_shapes
: *
valueB BOriginal

ImageSummaryImageSummaryImageSummary/tag	Reshape_5*
_output_shapes
: *

max_images*
	bad_colorB:џ  џ*
T0
h
Reshape_6/shapeConst*
dtype0*
_output_shapes
:*%
valueB"џџџџИ   И      
r
	Reshape_6ReshapexReshape_6/shape*1
_output_shapes
:џџџџџџџџџИИ*
T0*
Tshape0
И
Conv2D_3Conv2D	Reshape_6wc1/read*
paddingSAME*
data_formatNHWC*
strides
*/
_output_shapes
:џџџџџџџџџ..*
T0*
use_cudnn_on_gpu(
y
	BiasAdd_3BiasAddConv2D_3bc1/read*/
_output_shapes
:џџџџџџџџџ..*
data_formatNHWC*
T0
S
Relu_4Relu	BiasAdd_3*/
_output_shapes
:џџџџџџџџџ..*
T0
L
conv1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
W
conv1AddRelu_4conv1/y*/
_output_shapes
:џџџџџџџџџ..*
T0
h
Reshape_7/shapeConst*
dtype0*
_output_shapes
:*%
valueB"џџџџ.   .      
t
	Reshape_7Reshapeconv1Reshape_7/shape*/
_output_shapes
:џџџџџџџџџ..*
T0*
Tshape0
Y
ImageSummary_1/tagConst*
dtype0*
_output_shapes
: *
valueB B1.Conv

ImageSummary_1ImageSummaryImageSummary_1/tag	Reshape_7*
_output_shapes
: *

max_images*
	bad_colorB:џ  џ*
T0
Д
Conv2D_4Conv2Dconv1wc2/read*
paddingSAME*
data_formatNHWC*
strides
*/
_output_shapes
:џџџџџџџџџ..*
T0*
use_cudnn_on_gpu(
y
	BiasAdd_4BiasAddConv2D_4bc2/read*/
_output_shapes
:џџџџџџџџџ..*
data_formatNHWC*
T0
S
Relu_5Relu	BiasAdd_4*/
_output_shapes
:џџџџџџџџџ..*
T0
L
conv2/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
W
conv2AddRelu_5conv2/y*/
_output_shapes
:џџџџџџџџџ..*
T0
h
Reshape_8/shapeConst*
dtype0*
_output_shapes
:*%
valueB"џџџџ.   .      
t
	Reshape_8Reshapeconv2Reshape_8/shape*/
_output_shapes
:џџџџџџџџџ..*
T0*
Tshape0
Y
ImageSummary_2/tagConst*
dtype0*
_output_shapes
: *
valueB B2.Conv

ImageSummary_2ImageSummaryImageSummary_2/tag	Reshape_8*
_output_shapes
: *

max_images*
	bad_colorB:џ  џ*
T0
Д
Conv2D_5Conv2Dconv2wc3/read*
paddingSAME*
data_formatNHWC*
strides
*/
_output_shapes
:џџџџџџџџџ.. *
T0*
use_cudnn_on_gpu(
y
	BiasAdd_5BiasAddConv2D_5bc3/read*/
_output_shapes
:џџџџџџџџџ.. *
data_formatNHWC*
T0
S
Relu_6Relu	BiasAdd_5*/
_output_shapes
:џџџџџџџџџ.. *
T0
L
conv3/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
W
conv3AddRelu_6conv3/y*/
_output_shapes
:џџџџџџџџџ.. *
T0
h
Reshape_9/shapeConst*
dtype0*
_output_shapes
:*%
valueB"џџџџ.   .      
t
	Reshape_9Reshapeconv3Reshape_9/shape*/
_output_shapes
:џџџџџџџџџ..*
T0*
Tshape0
Y
ImageSummary_3/tagConst*
dtype0*
_output_shapes
: *
valueB B3.Conv

ImageSummary_3ImageSummaryImageSummary_3/tag	Reshape_9*
_output_shapes
: *

max_images *
	bad_colorB:џ  џ*
T0
e
HistogramSummary/tagConst*
dtype0*
_output_shapes
: *!
valueB BHistogram 1.Conv
e
HistogramSummaryHistogramSummaryHistogramSummary/tagwc1/read*
_output_shapes
: *
T0
e
HistogramSummary_1/tagConst*
dtype0*
_output_shapes
: *
valueB BHistogram pred
e
HistogramSummary_1HistogramSummaryHistogramSummary_1/tagpred*
_output_shapes
: *
T0
Ю
MergeSummary/MergeSummaryMergeSummaryScalarSummaryScalarSummary_1ImageSummaryImageSummary_1ImageSummary_2ImageSummary_3HistogramSummaryHistogramSummary_1*
_output_shapes
: *
N
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel

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
Љ
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesb_outbc1bc2bc3bd1w_outwc1wc2wc3wd1*
dtypes
2

}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
_class
loc:@save/Const*
T0
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

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2

save/AssignAssignb_outsave/RestoreV2*
use_locking(*
_output_shapes
:*
validate_shape(*
_class

loc:@b_out*
T0
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

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_1Assignbc1save/RestoreV2_1*
use_locking(*
_output_shapes
:*
validate_shape(*
_class

loc:@bc1*
T0
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

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_2Assignbc2save/RestoreV2_2*
use_locking(*
_output_shapes
:*
validate_shape(*
_class

loc:@bc2*
T0
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

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_3Assignbc3save/RestoreV2_3*
use_locking(*
_output_shapes
: *
validate_shape(*
_class

loc:@bc3*
T0
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

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_4Assignbd1save/RestoreV2_4*
use_locking(*
_output_shapes	
:ш*
validate_shape(*
_class

loc:@bd1*
T0
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

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_5Assignw_outsave/RestoreV2_5*
use_locking(*
_output_shapes
:	ш*
validate_shape(*
_class

loc:@w_out*
T0
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

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
 
save/Assign_6Assignwc1save/RestoreV2_6*
use_locking(*&
_output_shapes
:*
validate_shape(*
_class

loc:@wc1*
T0
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

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
 
save/Assign_7Assignwc2save/RestoreV2_7*
use_locking(*&
_output_shapes
:*
validate_shape(*
_class

loc:@wc2*
T0
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

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
 
save/Assign_8Assignwc3save/RestoreV2_8*
use_locking(*&
_output_shapes
: *
validate_shape(*
_class

loc:@wc3*
T0
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

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_9Assignwd1save/RestoreV2_9*
use_locking(* 
_output_shapes
:
	ш*
validate_shape(*
_class

loc:@wd1*
T0
Ж
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9

initNoOp^wc1/Assign^wc2/Assign^wc3/Assign^wd1/Assign^w_out/Assign^bc1/Assign^bc2/Assign^bc3/Assign^bd1/Assign^b_out/Assign""Ѕ
	summaries

ScalarSummary:0
ScalarSummary_1:0
ImageSummary:0
ImageSummary_1:0
ImageSummary_2:0
ImageSummary_3:0
HistogramSummary:0
HistogramSummary_1:0"ч
	variablesйж

wc1:0
wc1/Assign
wc1/read:0

wc2:0
wc2/Assign
wc2/read:0

wc3:0
wc3/Assign
wc3/read:0

wd1:0
wd1/Assign
wd1/read:0
%
w_out:0w_out/Assignw_out/read:0

bc1:0
bc1/Assign
bc1/read:0

bc2:0
bc2/Assign
bc2/read:0

bc3:0
bc3/Assign
bc3/read:0

bd1:0
bd1/Assign
bd1/read:0
%
b_out:0b_out/Assignb_out/read:0"
train_op

GradientDescent"ё
trainable_variablesйж

wc1:0
wc1/Assign
wc1/read:0

wc2:0
wc2/Assign
wc2/read:0

wc3:0
wc3/Assign
wc3/read:0

wd1:0
wd1/Assign
wd1/read:0
%
w_out:0w_out/Assignw_out/read:0

bc1:0
bc1/Assign
bc1/read:0

bc2:0
bc2/Assign
bc2/read:0

bc3:0
bc3/Assign
bc3/read:0

bd1:0
bd1/Assign
bd1/read:0
%
b_out:0b_out/Assignb_out/read:0ЂA
//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

var CATS = {
	TYPE :0,		// MLP, Conv
	ACT :1,			// activation function type
	CHANNEL :2,
	SIZE :3,		// input size (ch * x * y)
	XSIZE :4,		// width
	YSIZE :5,		// height
	KSIZE :6,		// kernel size
	STRIDE :7,
	LPLEN :8,		// length of layer params
};

// activation function
var act = {};
act[0] = function(x) {	// identity
	return x;
};
act[2] = function(x) {	// sigmoid
	return 1/(1+Math.exp(-x));
};
act[3] = function(x) {	// tanh
	return tanh(x);
};
act[5] = function(x) {	// ReLU
	return (x>0 ? x : 0);
};
act[6] = function(x) {	// LeakyReLU
	return (x>0 ? x : x*0.01);
};

var layer_forward = {};
layer_forward[0] = function(s, w, o, uu, u)	// linear
{
	var s_in = uu[CATS.SIZE]+1;	// +1 -> for bias
	var s_out = u[CATS.SIZE];
	var a = u[CATS.ACT];
	console.log("mlp:" + s_in +" "+ s_out +" "+ a);

	for (var i=0; i<s_out; i++) {
		var z = 0;
		for (var j=0; j<s_in; j++) {
			z += w[j*s_out+i]*s[j];
		}
		o[i] = act[a](z);
	}
};
layer_forward[1] = function(s, w, o, uu, u)	// convolutional
{
	var sx = u[CATS.XSIZE] - Math.floor(u[CATS.KSIZE]/2)*2;	// out
	var sy = u[CATS.YSIZE] - Math.floor(u[CATS.KSIZE]/2)*2;
	var n = u[CATS.CHANNEL] * u[CATS.KSIZE]*u[CATS.KSIZE];
	var size = uu[CATS.SIZE]/uu[CATS.CHANNEL];
	var step = u[CATS.XSIZE]-u[CATS.KSIZE];
	var a = u[CATS.ACT];

	console.log(sx +" "+ sy +" "+ a);
	//console.log(s);

	var m = 0;
	for (var c=0; c<u[CATS.CHANNEL]; c++) {	// out
		for (var y=0; y<sy; y++) {
			for (var x=0; x<sx; x++) {
				var z = 0;
				for (var cc=0; cc<uu[CATS.CHANNEL]; cc++) {	// in
					var k = c*(u[CATS.KSIZE]*u[CATS.KSIZE]) + cc*n;
					var p = (size*cc) + y*u[CATS.XSIZE]+x;	// in
					for (var wy=u[CATS.KSIZE]; wy>0; wy--) {
						for (var wx=u[CATS.KSIZE]; wx>0; wx--) {
							z += s[p++] * w[k++];
						}
						p += step;
					}
				}
				o[m++] = act[a](z);
			}
		}
	}
	console.log("min:"+Math.min.apply(null, o) +" max:"+ Math.max.apply(null, o));
};
function _CatsEye(w, u)
{
	this.w = w;
	this.u = u;

	var n = 0;
	this.o = [];
	for (var uu in u) {
		this.o[n++] = [];
	}
}
_CatsEye.prototype = {
	// caluculate forward propagation of input x
	forward: function(x)
	{
		// calculation of input layer
		//this.o[0][u[0][CATS.SIZE]] = 1;
		x[u[0][CATS.SIZE]] = 1;

		layer_forward[u[1][CATS.TYPE]](/*this.o[0]*/x, this.w[0], this.o[1], u[0], u[1]);
		for (var i=1; i<u.length-1; i++) {
			this.o[i][u[i][CATS.SIZE]] = 1;
			layer_forward[u[i+1][CATS.TYPE]](this.o[i], this.w[i], this.o[i+1], u[i], u[i+1]);
		}
	},

	predict: function(x)
	{
		// forward propagation
		this.forward(x);
		// biggest output means most probable label
		var n = u.length-1;
		var max = this.o[n][0];
		var ans = 0;
		for (var i=1; i<u[n][CATS.SIZE]; i++) {
			console.log("ans:"+this.o[n][i]);
			if (this.o[n][i] > max) {
				max = this.o[n][i];
				ans = i;
			}
		}
		for (i=0; i<u[n][CATS.SIZE]; i++) this.o[n][i] /= max;
		return ans;
	}
};


function CatsEye(_in, _hid, _out, w1, w2) {
	this.in = _in;
	this.hid = _hid;
	this.out = _out;

	// input layer
	this.xi1 = [];
	this.xi2 = [];
	this.xi3 = [];
	// output layer
	this.o1 = [];
	this.o2 = [];
	this.o3 = [];
	// error value
	this.d2 = [];
	this.d3 = [];
	// weights
	this.w1 = w1;
	this.w2 = w2;
};
CatsEye.prototype = {
	// activation function
	sigmoid: function(x)
	{
		return 1/(1+Math.exp(-x));
	},

	// caluculate forward propagation of input x
	forward: function(x)
	{
		// calculation of input layer
		x[this.in] = 1;

		// caluculation of hidden layer
		for (j=0; j<this.hid; j++) {
			this.xi2[j] = 0;
			for (i=0; i<this.in+1; i++) {
				this.xi2[j] += this.w1[i*this.hid+j]*x[i];
			}
			this.o2[j] = this.sigmoid(this.xi2[j]);
		}
		this.o2[this.hid] = 1;

		// caluculation of output layer
		for (j=0; j<this.out; j++) {
			this.xi3[j] = 0;
			for (i=0; i<this.hid+1; i++) {
				this.xi3[j] += this.w2[i*this.out+j]*this.o2[i];
			}
			this.o3[j] = this.xi3[j];
			//this.o3[j] = this.sigmoid(this.xi3[j]);
		}
	},

	predict: function(x)
	{
		// forward propagation
		this.forward(x);
		// biggest output means most probable label
		max = this.o3[0];
		ans = 0;
		for (i=1; i<this.out; i++) {
			if (this.o3[i] > max) {
				max = this.o3[i];
				ans = i;
			}
		}
		for (i=0; i<this.out; i++) this.o3[i] /= max;
		return ans;
	}
};

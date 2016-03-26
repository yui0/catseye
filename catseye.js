//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

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
	// sigmoid function
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
			this.o3[j] = this.sigmoid(this.xi3[j]);
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

// calculate forward propagation
/*void _CatsEye_convolutional_forward(CatsEye_layer *l)
{
	int step = l->sx - l->ksize;
///	int ks = l->ksize * l->ksize;

	// c[out], c[in], ksize, h, w
	real *s, *z, *o;
	real *w = l->W;
	memset(l->z, 0, sizeof(real)*l->ch*l->ox*l->oy);
	for (int c=0; c<l->ch; c++) {	// out
		real *r = s = l->x;
		o = l->z + c*l->ox*l->oy;
		for (int cc=l->ich; cc>0; cc--) {	// in
//			s = l->x + (l->ich-cc)*l->sx*l->sy;
			for (int wy=0; wy<l->ksize; wy++) {
				for (int wx=0; wx<l->ksize; wx++) {
					real *p = s++;	// in
					z = o;		// out
///					real W = w[wx*l->ksize+wy];
					for (int y=l->py; y>0; y--) {
						_fma(z, p, *w, l->px);	// *z++ += (*p++) * (*w); p += m;
///						_fma(z, p, W, l->px);	// *z++ += (*p++) * (*w); p += m;
						p += l->sx;
						z += l->ox;
					}
					w++;
				}
				s += step;
////				s += l->sx;
			}
			r += l->sx * l->sy;
			s = r;
///			w += ks;
		}
	}
}*/
// calculate back propagation
/*void _CatsEye_convolutional_backward(CatsEye_layer *l)
{
	int step = l->sx - l->ksize;
///	int ks = l->ksize * l->ksize;

	// c[out], c[in], ksize, h, w
	real *d, *prev_delta;
	real *w = l->W;
	real *delta = l->dOut;
	memset(l->dIn, 0, sizeof(real)*l->ich*l->sx*l->sy);
	for (int c=l->ch; c>0; c--) {	// out
		real *r = prev_delta = l->dIn;
		for (int cc=l->ich; cc>0; cc--) {	// in
			for (int wy=0; wy<l->ksize; wy++) {
				for (int wx=0; wx<l->ksize; wx++) {
					real *p = prev_delta++;	// in
					d = delta;		// out
///					real W = w[wx*l->ksize+wy];
					for (int y=l->oy; y>0; y--) {
						_fma(p, d, *w, l->ox);	// *p++ += (*d++) * (*w);
///						_fma(p, d, W, l->ox);	// *p++ += (*d++) * (*w);
						p += l->sx;
						d += l->ox;
					}
					w++;
				}
				prev_delta += step;
			}
			r += l->sx * l->sy;
			prev_delta = r;
///			w += ks;
		}
		delta = d;
//		prev_delta = l->dIn;
	}
}*/
// update the weights
/*void _CatsEye_convolutional_update(CatsEye_layer *l)
{
	int step = l->sx - l->ksize;
///	int ks = l->ksize * l->ksize;

	// c[out], c[in], ksize, h, w
	real *w = l->W;
	real *d, *prev_out;
	real *delta = l->dOut;
	for (int c=l->ch; c>0; c--) {	// out
		real *r = prev_out = l->x;
		for (int cc=l->ich; cc>0; cc--) {	// in
			for (int wy=0; wy<l->ksize; wy++) {
				for (int wx=0; wx<l->ksize; wx++) {
					real *p = prev_out++;	// in
					d = delta;		// out
///					real dOut = delta[wx*l->ksize+wy];
					register real a = 0;
					for (int y=l->oy; y>0; y--) {
						a += dot(d, p, l->ox);	// a += d * p;
///						a += dot(dOut, p, l->ox);	// a += d * p;
						p += l->sx;
						d += l->ox;
					}
					*w++ += -l->eta * a;
				}
				prev_out += step;
			}
			r += l->sx * l->sy;
			prev_out = r;
		}
		delta = d;
//		prev_out = l->x;
	}
}*/


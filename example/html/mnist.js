var cat;

function Main() {
        cat = new CatsEye(config[0], config[1], config[2], w1, w2);

        this.input = document.getElementById('input');
        this.canvas = document.getElementById('sketch');
        this.canvas.width  = 225; // 8 * 28 + 1
        this.canvas.height = 225; // 8 * 28 + 1
        this.ctx = this.canvas.getContext('2d');
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mouseup',   this.onMouseUp.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));

        this.canvas.addEventListener('touchstart', this.onMouseDown.bind(this));
        this.canvas.addEventListener('touchend',   this.onMouseUp.bind(this));
        this.canvas.addEventListener('touchmove', this.onMouseMove.bind(this));
        this.initialize();
    };
Main.prototype = {
    initialize: function() {
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fillRect(0, 0, 225, 225);
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(0, 0, 225, 225);
        this.ctx.lineWidth = 0.05;
        for (var i = 0; i < 27; i++) {
            this.ctx.beginPath();
            this.ctx.moveTo((i + 1) * 8,   0);
            this.ctx.lineTo((i + 1) * 8, 225);
            this.ctx.closePath();
            this.ctx.stroke();

            this.ctx.beginPath();
            this.ctx.moveTo(  0, (i + 1) * 8);
            this.ctx.lineTo(225, (i + 1) * 8);
            this.ctx.closePath();
            this.ctx.stroke();
        }
        this.drawInput();
        $('#output td').text('').removeClass('success');
    },
    onMouseDown: function(e) {
        this.canvas.style.cursor = 'default';
        this.drawing = true;
        this.prev = this.getPosition(e);
    },
    onMouseUp: function() {
        this.drawing = false;
        this.drawInput();
    },
    onMouseMove: function(e) {
        if (this.drawing) {
            var curr = this.getPosition(e);
            this.ctx.lineWidth = 16;
            this.ctx.lineCap = 'round';
            this.ctx.beginPath();
            this.ctx.moveTo(this.prev.x, this.prev.y);
            this.ctx.lineTo(curr.x, curr.y);
            this.ctx.stroke();
            this.ctx.closePath();
            this.prev = curr;
            //$('#output tr').eq(2).find('td').eq(0).text(curr.x+" "+curr.y);
        }
    },
    getPosition: function(e) {
        var rect = this.canvas.getBoundingClientRect();

	if (e.changedTouches) e = e.changedTouches[0];
        //if (!e.pageX) e = event.touches[0];
        //$('#output tr').eq(1).find('td').eq(0).text(e.pageX+" "+e.pageY);

        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    },
    drawInput: function() {
        var ctx = this.input.getContext('2d');
        var img = new Image();
        img.onload = function() {
            var inputs = [];
            var small = document.createElement('canvas').getContext('2d');
            small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28);
            var data = small.getImageData(0, 0, 28, 28).data;
            for (var i = 0; i < 28; i++) {
                for (var j = 0; j < 28; j++) {
                    var n = 4 * (i * 28 + j);
                    inputs[i * 28 + j] = (data[n + 0] + data[n + 1] + data[n + 2]) / 3;
                    ctx.fillStyle = 'rgb(' + [data[n + 0], data[n + 1], data[n + 2]].join(',') + ')';
                    ctx.fillRect(j * 5, i * 5, 5, 5);
                }
            }
            /*if (Math.min(...inputs) === 255) {
                return;
            }*/
            if (Math.min.apply(null, inputs) === 255) return;
            /*for (n in inputs) {
                if (n === 255)
            }*/
            for (i=0; i<28*28; i++) inputs[i] = 255-inputs[i];
            for (i=0; i<28*28; i++) inputs[i] /= 255;
            a = cat.predict(inputs);
            //$('#output tr').eq(a+1).find('td').eq(0).text(a);
            for (i=0; i<10; i++) {
                $('#output tr').eq(i+1).find('td').eq(0).text(cat.o3[i].toFixed(2));
                if (a === i) {
                    $('#output tr').eq(i+1).find('td').eq(0).addClass('success');
                } else {
                    $('#output tr').eq(i+1).find('td').eq(0).removeClass('success');
                }
            }
        };
        img.src = this.canvas.toDataURL();
    }
}

$(document).ready(function() {
    // prevent elastic scrolling (for mobile)
    document.body.addEventListener('touchmove', function (event) {
        event.preventDefault();
    }, false); // end body.onTouchMove

    var main = new Main();
    $('#clear').click(function() {
        main.initialize();
    });
});

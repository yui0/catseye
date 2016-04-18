var cat;

function Main() {
        cat = new _CatsEye(w, u);

        this.input = document.getElementById('input');
        this.canvas = document.getElementById('sketch');
        this.canvas.width  = 257; // 8 * 32 + 1
        this.canvas.height = 257; // 8 * 32 + 1
        this.ctx = this.canvas.getContext('2d');

        this.initialize();
    };
Main.prototype = {
    initialize: function() {
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fillRect(0, 0, 257, 257);
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(0, 0, 257, 257);
        this.ctx.lineWidth = 0.05;
//        this.drawInput();
        $('#output td').text('').removeClass('success');
    },
    drawInput: function() {
        var ctx = this.input.getContext('2d');
        var img = new Image();
        img.onload = function() {
            var inputs = [];
            var small = document.createElement('canvas').getContext('2d');
            small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 32, 32);
            var data = small.getImageData(0, 0, 32, 32).data;
            for (var i = 0; i < 32; i++) {
                for (var j = 0; j < 32; j++) {
                    var n = 4 * (i * 32 + j);
                    //inputs[i * 32 + j] = (data[n + 0] + data[n + 1] + data[n + 2]) / 3;
                    inputs[(i*32+j)*3  ] = data[n];
                    inputs[(i*32+j)*3+1] = data[n+1];
                    inputs[(i*32+j)*3+2] = data[n+2];
                    ctx.fillStyle = 'rgb(' + [data[n + 0], data[n + 1], data[n + 2]].join(',') + ')';
                    ctx.fillRect(j * 5, i * 5, 5, 5);
                }
            }
            //for (i=0; i<32*32; i++) inputs[i] = 255-inputs[i];
            for (i=0; i<32*32*3; i++) inputs[i] /= 255;
            a = cat.predict(inputs);
            //$('#output tr').eq(a+1).find('td').eq(0).text(a);
            for (i=0; i<10; i++) {
                $('#output tr').eq(i+1).find('td').eq(0).text(cat.o[u.length-1][i].toFixed(2));
                if (a === i) {
                    $('#output tr').eq(i+1).find('td').eq(0).addClass('success');
                } else {
                    $('#output tr').eq(i+1).find('td').eq(0).removeClass('success');
                }
            }
        };
        img.src = this.canvas.toDataURL();
    },

    readURL: function(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            var image = new Image();

            reader.onload = function(e) {
                image.onload = function() {
                    canvas = document.getElementById('sketch');
                    ctx = canvas.getContext('2d');
                    ctx.drawImage(image, 0, 0, 257, 257);
                }
                image.src = e.target.result;
            }

            reader.readAsDataURL(input.files[0]);
        }
    }
}

$(document).ready(function() {
    // prevent elastic scrolling (for mobile)
    document.body.addEventListener('touchmove', function (event) {
        event.preventDefault();
    }, false); // end body.onTouchMove

    var main = new Main();
    $("img.iClick").click(function(){
        /*var imgSrc = $(this).attr("src");
        var imgAlt = $(this).attr("alt");
        $("img#MainPhoto").attr({src:imgSrc, alt:imgAlt});
        $("img#MainPhoto").hide();
        $("img#MainPhoto").fadeIn("slow");*/

        canvas = document.getElementById('sketch');
        ctx = canvas.getContext('2d');
        //ctx.drawImage(this, 0, 0);
        ctx.drawImage(this, 0, 0, this.width, this.height, 0, 0, 257, 257);
        return false;
    });
    $('#recognize').click(function() {
        main.drawInput();
    });
    $("#img").change(function(){
        main.readURL(this);
    });
});

/**
 * Created by Mohammed Alaa on 5/24/2017.
 */
var canvas, ctx, wd, he, rotMax;

function placeWords(words) {
    //placement algorithm
    ctx.clearRect(0, 0, wd, he);

    var tries = 100;

    while (true) {
        var rects = [],
            rect, len, shW, shH, arg, delx, dely, maxiter = 5000;

        for (var ind in words) {

            ctx.fillStyle = "rgb(" + (Math.floor(Math.random() * 255)) + "," + (Math.floor(Math.random() * 255)) + "," + (Math.floor(Math.random() * 255)) + ")";

            ctx.font = 'bold ' + (65 - .5 * (ind + 1)) + 'px Monaco'; // size /1.38888888889 = width of char
            var ch = (65 - .5 * (ind + 1)),
                cw = ch / 1.38888888889;

            len = words[ind].length * cw;

            ctx.save();

            do {

                arg = (Math.floor(Math.random() * 2 * rotMax) - rotMax) * Math.PI / 180;


                delx = len / 6 * Math.sin(arg);
                dely = len / 6 * Math.cos(arg);

                shW = len / 2 + (Math.floor(Math.random() * wd)) % (wd - len);
                shH = ch / 2 + Math.floor(Math.random() * he) % (wd - ch);

                rect = {
                    BL: {
                        x: shW - len / 2 - Math.abs(delx) * 3,
                        y: shH + ch / 2 + dely * 1.5
                    },
                    TR: {
                        x: shW + len / 2 + Math.abs(delx) * 3,
                        y: shH - ch / 2 - dely * 1.5
                    }
                };

                maxiter--;
            } while (maxiter > 0 && (isIntersectedWithAny(rect, rects) || !isInDomain(rect)));
            ctx.translate(shW, shH);
            ctx.rotate(arg);
            ctx.fillText(words[ind], 0, 0);
            ctx.restore();

            rects.push(rect)
        }
        //console.log(maxiter)
        if (maxiter <= 0) {
            console.error("Sorry I failed to layout but i will try again,I can do more ", tries, " tries");
            ctx.clearRect(0, 0, wd, he);
        }
        if (tries <= 0) {
            console.log("Sorry I Can't try any more");
            break;
        }

        if (maxiter > 0) {
            console.log("I managed to succeed in", 100 - tries);

            break;
        }

        tries--;
    }
    $("#canvas").animate({opacity:1});
}
$(document).ready(function() {
    canvas = $("#canvas")[0]
    ctx = canvas.getContext("2d");
    wd = canvas.width;
    he = canvas.height;
    rotMax = 22;

    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    $("#send_btn").click(function() {
        $("#canvas").animate({opacity:0})
        setTimeout(function() {

            if ($('#query_txt').val() != '') {
                $("#send_btn").prop('disabled', true);

                $.post("/", {
                    'query': $('#query_txt').val(),
                    'model': ($('#model_ck')[0].checked) ? 0 : 1
                }, function(words) {
                    placeWords(words)
                    $("#send_btn").prop('disabled', false);
                }).fail(function(xhr, status, error) {

                    placeWords(["Can't connect", "Server error", "No connection", "Connection refused", "No response", "معلش", "معلش"]);
                    $("#send_btn").prop('disabled', false);
                });;

            } else
                placeWords(["No input", "Query?", "Enter Word", "Human?", "معلش", "متضغطتش على زراير و انت مش فاهم", "معلش"]);



        }, 550);


    });
});


function isIntersected(rectA, rectB) {
    return !(rectA.BL.x > rectB.TR.x || rectA.BL.y < rectB.TR.y ||
        rectA.TR.x < rectB.BL.x || rectA.TR.y > rectB.BL.y);
}

function isIntersectedWithAny(rectA, rects) {
    for (var ind in rects)
        if (isIntersected(rectA, rects[ind]))
            return true;
    return false;
}
var wd, he;

function isInDomain(rect) {
    return (rect.BL.x > 0 && rect.TR.x < wd && rect.BL.y < he && rect.TR.y > 0);
}

function isValidKey(evt,trigger){
    var charCode = (evt.which) ? evt.which : event.keyCode;
    if(String.fromCharCode(charCode) == ' ' ||String.fromCharCode(charCode) == '\n' ||String.fromCharCode(charCode) == '\t' )
        return false;

    return true;


}
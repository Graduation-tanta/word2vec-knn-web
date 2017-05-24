/**
 * Created by Mohammed Alaa on 5/24/2017.
 */
function isIntersected(rectA, rectB) {
    return !(rectA.BL.x > rectB.TR.x || rectA.BL.y < rectB.TR.y
    || rectA.TR.x < rectB.BL.x || rectA.TR.y > rectB.BL.y);
}

function isIntersectedWithAny(rectA, rects) {
    for (var ind in rects)
        if (isIntersected(rectA, rects[ind]))
            return true;
    return false;
}
var wd , he ;
function isInDomain(rect) {
    return (rect.BL.x > 0 && rect.TR.x < wd && rect.BL.y < he && rect.TR.y > 0);
}

//placement algorithm
$(function () {
    words = ['AAAAAAA','AAAAAAAAA','AAAAAAAAA','AAAAAAAAAAAAAAAAA','AAAAAAAAA','AAAAAAAAA','AAAAAAAAA'];

    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    wd = canvas.width; he = canvas.height ; rotMax = 20;

    ctx.textAlign = "center";
    ctx.textBaseline = "middle";



    var tries=15;

    while (true){
        var rects = [], rect, len, shW, shH, arg, delx, dely,maxiter=5000;

        for (var ind in words) {

            ctx.fillStyle = "rgb("+(Math.floor(Math.random() * 255))+","+(Math.floor(Math.random() * 255))+","+(Math.floor(Math.random() * 255))+")";

            ctx.font = 'bold ' + (20+.3*(ind+1)) + 'px Monaco'; // size /1.38888888889 = width of char
            var ch = (20+.3*(ind+1)), cw = ch / 1.38888888889;

            len = words[ind].length * cw;

            ctx.save();

            do {

                arg = (Math.floor(Math.random() * 2 * rotMax) - rotMax) * Math.PI / 180;

                delx =len / 2 * Math.sin(arg);
                dely =len / 2 * Math.cos(arg);

                shW = len / 2  + (Math.floor(Math.random() * wd)) % (wd - len);
                shH = ch / 2  + Math.floor(Math.random() * he) % (wd - ch);

                rect = {
                    BL: {x: shW - len / 2+delx, y: shH + ch / 2 + dely},
                    TR: {x: shW + len / 2-delx, y: shH - ch / 2 - dely}
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
        if(maxiter <= 0)
        { console.error("Sorry I failed to layout but i will try again,I can do more ",tries," tries");
            ctx.clearRect(0, 0,wd,he);
        }
        if(tries <=0 ) {
            console.log("Sorry I Can't try any more");
            break;
        }

        if(maxiter >0)
        {   console.log("I managed to succeed");
            break;
        }

        tries--;
    }
}); // end $(function(){});
"""Export an Everything Cubed plan as a standalone interactive HTML view and an OBJ mesh."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def mesh_preview_from_ply(path, max_faces=250000):
    """Embed colored vertices for WebGL, preserving per-vertex RGB interpolation."""
    if not path or not Path(path).is_file():
        return None
    import numpy as np
    from mesh_export import VERTEX, FACE
    with Path(path).open("rb") as stream:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = stream.readline()
            if not line or len(header) > 4096:
                raise ValueError("invalid PLY header")
            header += line
        text = header.decode("ascii")
        if "format binary_little_endian 1.0" not in text:
            raise ValueError("only our binary PLY files can be previewed")
        nv = int(next(s.split()[2] for s in text.splitlines() if s.startswith("element vertex ")))
        nf = int(next(s.split()[2] for s in text.splitlines() if s.startswith("element face ")))
        vertices = np.fromfile(stream, dtype=VERTEX, count=nv)
        faces = np.fromfile(stream, dtype=FACE, count=nf)
    if len(vertices) != nv or len(faces) != nf:
        raise ValueError("truncated PLY mesh")
    step = max(1, int(np.ceil(nf / max_faces)))
    chosen = faces[::step]
    ids = np.stack([chosen["a"], chosen["b"], chosen["c"]], axis=1).ravel()
    packed = np.empty(len(ids), dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                               ("r", "u1"), ("g", "u1"), ("b", "u1"), ("pad", "u1")]))
    for target, source in (("x", "x"), ("y", "y"), ("z", "z"),
                           ("r", "red"), ("g", "green"), ("b", "blue")):
        packed[target] = vertices[source][ids]
    packed["pad"] = 0
    bounds = [[float(np.min(packed[k])) for k in ("x", "y", "z")],
              [float(np.max(packed[k])) for k in ("x", "y", "z")]]
    return {"bounds": bounds, "vertices_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
            "preview_faces": len(chosen), "total_faces": nf}


COLORS = {"white": "#e8e6e1", "light_gray": "#b7b8b3", "gray": "#747978", "black": "#303137",
          "brown": "#805538", "red": "#b9493d", "orange": "#d78747", "yellow": "#e7c45c",
          "lime": "#86b950", "green": "#638455", "cyan": "#5cabb6", "light_blue": "#8eafcf",
          "blue": "#526fa7", "purple": "#8b70a6", "magenta": "#ac73a7", "pink": "#d9a8b4"}
WOOD = {"oak": "#a77d50", "spruce": "#765537", "birch": "#dac596", "jungle": "#aa754e",
        "acacia": "#b36f49", "dark_oak": "#56422f", "mangrove": "#85473e", "cherry": "#d39b91"}


def material_color(name):
    name = str(name).removeprefix("minecraft:")
    for key, color in WOOD.items():
        if name.startswith(key + "_"):
            return color
    for key, color in COLORS.items():
        if name.startswith(key + "_"):
            return color
    if "glass" in name: return "#9bd6df"
    if "stone" in name or "andesite" in name: return "#92948e"
    if "brick" in name: return "#986154"
    if "iron" in name or "metal" in name: return "#a9b5bc"
    return "#b7aaa0"


def scene_from_plan(plan):
    """A compact box model in plan coordinates; roof omitted so the room stays visible."""
    size = plan["size"]
    W, D, H = (int(size[k]) for k in ("x", "z", "y"))
    boxes = []
    footprint = plan.get("footprint") or [[1] * W for _ in range(D)]
    inside = lambda x, z: 0 <= x < W and 0 <= z < D and bool(footprint[z][x])

    def add(x, y, z, w, h, d, color, kind, label):
        if min(w, h, d) > 0:
            boxes.append([x, y, z, w, h, d, color, kind, label])

    floor = plan.get("floor", {})
    fg, fp = floor.get("grid", []), floor.get("palette", [])
    for z in range(D):
        for x in range(W):
            if not inside(x, z):
                continue
            mat = fp[fg[z][x]] if z < len(fg) and x < len(fg[z]) and fp else "oak_planks"
            add(x, -0.12, z, 1, 0.12, 1, material_color(mat), "floor", "Floor")

    openings = plan.get("openings", [])
    for side in ("north", "south", "west", "east"):
        wall = plan.get("walls", {}).get(side, {})
        grid, palette = wall.get("grid", []), wall.get("palette", [])
        length = W if side in ("north", "south") else D
        for y in range(H):
            for u in range(length):
                opening = next((o for o in openings if o.get("side") == side and
                                o.get("u", -1) <= u < o.get("u", -1) + o.get("w", 1) and
                                o.get("y", -1) <= y < o.get("y", -1) + o.get("h", 1) and
                                o.get("type") in ("window", "door")), None)
                if opening and opening["type"] == "door": continue
                mat = (palette[grid[y][u]] if y < len(grid) and u < len(grid[y]) and palette
                       else plan.get("wall_main", "white_concrete"))
                color = "#9bd6df" if opening else material_color(mat)
                x, z, w, d = {"north": (u, -0.08, 1, 0.08), "south": (u, D, 1, 0.08),
                              "west": (-0.08, u, 0.08, 1), "east": (W, u, 0.08, 1)}[side]
                ix, iz = {"north": (u, 0), "south": (u, D - 1),
                          "west": (0, u), "east": (W - 1, u)}[side]
                if not inside(ix, iz):
                    continue
                add(x, y, z, w, 1, d, color, "opening" if opening else "wall",
                    opening["type"].title() if opening else "Wall")

    for z in range(D):
        for x in range(W):
            if not inside(x, z):
                continue
            for nx, nz, bx, bz, bw, bd in ((x + 1, z, x + 1, z, .08, 1),
                                           (x - 1, z, x - .08, z, .08, 1),
                                           (x, z + 1, x, z + 1, 1, .08),
                                           (x, z - 1, x, z - .08, 1, .08)):
                if 0 <= nx < W and 0 <= nz < D and not inside(nx, nz):
                    add(bx, 0, bz, bw, H, bd, material_color(plan.get("wall_main", "white_concrete")), "wall", "Wall")

    for o in plan.get("objects", []):
        typ = o.get("type", "object")
        x, z = float(o.get("x", 0)), float(o.get("z", 0))
        w, d = max(0.3, float(o.get("w", 1))), max(0.3, float(o.get("d", 1)))
        h = max(0.2, float(o.get("h", 1)))
        y = float(o.get("y", 1)) - 1 if typ in ("monitor", "laptop", "book_stack", "mug", "small_plant",
                                                 "printer", "desk_lamp", "projector", "chandelier", "ceiling_fan",
                                                 "linear_light", "pendant_lamp", "led_panel", "downlight",
                                                 "smoke_detector", "ceiling_light") else 0
        if typ in ("monitor", "laptop", "book_stack", "mug", "small_plant", "printer", "desk_lamp"):
            h, w, d = 0.45, min(w, 0.7), min(d, 0.7)
        if typ in ("led_panel", "downlight", "smoke_detector", "ceiling_light"):
            y, h = H - 0.12, 0.1
        if typ in ("linear_light", "pendant_lamp", "chandelier"):
            h = min(h, H - y)
        if typ == "rug": h = 0.04
        color = (COLORS.get(o.get("color"), "") or WOOD.get(o.get("material"), "") or
                 {"monitor": "#272e37", "laptop": "#3a4959", "fire_extinguisher": "#c84339",
                  "plant": "#648959", "small_plant": "#648959", "led_panel": "#f2e9c8",
                  "downlight": "#f2e9c8", "linear_light": "#f2e9c8", "pendant_lamp": "#e8d9a8"}.get(typ, "#a78465"))
        add(x, y, z, w, h, d, color, "object", o.get("name", typ.replace("_", " ").title()))

    for o in openings:
        typ = o.get("type")
        if typ in ("door", "window"): continue
        side, u = o.get("side"), o.get("u", 0)
        y, h, width = o.get("y", 0), max(0.2, o.get("h", 1)), max(0.3, o.get("w", 1))
        x, z, w, d = {"north": (u, 0.02, width, 0.08), "south": (u, D - 0.1, width, 0.08),
                      "west": (0.02, u, 0.08, width), "east": (W - 0.1, u, 0.08, width)}[side]
        add(x, y, z, w, h, d, "#c63d35" if typ == "fire_alarm" else "#ddd4c4",
            "object", typ.replace("_", " ").title())
    return {"size": [W, H, D], "boxes": boxes, "source": plan.get("source", {})}


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Everything Cubed · 3D room</title><style>
*{box-sizing:border-box}body{margin:0;background:#111820;color:#edf2f2;font:14px system-ui,sans-serif;overflow:hidden}
canvas{display:block;width:100vw;height:100vh;cursor:grab;position:absolute;inset:0}canvas:active{cursor:grabbing}
.panel{position:fixed;left:18px;top:18px;background:#17232ce8;border:1px solid #39505a;border-radius:12px;padding:16px;max-width:300px;box-shadow:0 8px 30px #0007}
h1{font-size:17px;margin:0 0 7px}.muted{color:#acc0c7;font-size:12px;line-height:1.5}label{display:block;margin-top:12px}
input[type=range]{width:100%}button,a{margin-top:9px;border:1px solid #5a7480;background:#243945;color:#eff;border-radius:7px;padding:7px 10px;cursor:pointer;text-decoration:none;display:inline-block}
#tip{position:fixed;pointer-events:none;background:#0d151bdb;padding:5px 8px;border-radius:6px;display:none}
</style></head><body><canvas id="view"></canvas><canvas id="meshview" style="display:none"></canvas><div class="panel"><h1>Everything Cubed · 3D room</h1>
<div id="stats" class="muted"></div><div class="muted">Drag to orbit · scroll to zoom · hover for names</div>
<label>View <select id="mode"><option value="plan">Room model</option><option value="mesh">Reconstructed surfaces</option></select></label>
<label><input id="walls" type="checkbox" checked> Show walls</label><label><input id="objects" type="checkbox" checked> Show objects</label>
<label>Wall opacity <input id="alpha" type="range" min="0" max="100" value="40"></label>
<button id="reset">Reset view</button> <a id="photoLink" href="photo_view.html" style="display:none">Photo textures</a></div><div id="tip"></div><script>
const scene = __SCENE__;
if(scene.photo_view)document.getElementById('photoLink').style.display='inline-block';
const canvas=document.getElementById('view'),ctx=canvas.getContext('2d'),tip=document.getElementById('tip');
const meshCanvas=document.getElementById('meshview');let gl=null,meshProgram=null,meshCount=0;
function shader(type,source){let s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}
function initMesh(){if(!scene.mesh)return;gl=meshCanvas.getContext('webgl',{antialias:true});if(!gl){mode.options[1].disabled=true;return}
 const vs=shader(gl.VERTEX_SHADER,`attribute vec3 aPos;attribute vec3 aColor;varying vec3 vColor;uniform vec3 uCenter;uniform float uYaw,uPitch,uZoom,uAspect;
 void main(){vec3 p=aPos-uCenter;float c=cos(uYaw),s=sin(uYaw);float a=p.x*c-p.z*s,b=p.x*s+p.z*c;float q=b*cos(uPitch)-p.y*sin(uPitch),v=b*sin(uPitch)+p.y*cos(uPitch);float depth=max(0.1,uZoom-q);gl_Position=vec4(a/uAspect,v,depth*1.0002-0.2,depth);vColor=aColor;}`);
 const fs=shader(gl.FRAGMENT_SHADER,`precision mediump float;varying vec3 vColor;void main(){gl_FragColor=vec4(vColor,1.0);}`);
 meshProgram=gl.createProgram();gl.attachShader(meshProgram,vs);gl.attachShader(meshProgram,fs);gl.linkProgram(meshProgram);
 if(!gl.getProgramParameter(meshProgram,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(meshProgram));gl.useProgram(meshProgram);
 let raw=atob(scene.mesh.vertices_b64),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
 let buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,bytes,gl.STATIC_DRAW);meshCount=bytes.length/16;
 let pos=gl.getAttribLocation(meshProgram,'aPos'),color=gl.getAttribLocation(meshProgram,'aColor');gl.enableVertexAttribArray(pos);gl.vertexAttribPointer(pos,3,gl.FLOAT,false,16,0);
 gl.enableVertexAttribArray(color);gl.vertexAttribPointer(color,3,gl.UNSIGNED_BYTE,true,16,12);gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);
}
function drawMesh(){if(!gl)return;gl.viewport(0,0,meshCanvas.width,meshCanvas.height);gl.clearColor(.067,.094,.125,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
 gl.uniform3fv(gl.getUniformLocation(meshProgram,'uCenter'),center);gl.uniform1f(gl.getUniformLocation(meshProgram,'uYaw'),yaw);
 gl.uniform1f(gl.getUniformLocation(meshProgram,'uPitch'),pitch);gl.uniform1f(gl.getUniformLocation(meshProgram,'uZoom'),zoom);
 gl.uniform1f(gl.getUniformLocation(meshProgram,'uAspect'),innerWidth/innerHeight);gl.drawArrays(gl.TRIANGLES,0,meshCount)}
const [W,H,D]=scene.size;let yaw=-.65,pitch=.55,zoom=Math.max(W,D,H)*1.7,drag=false,last=[0,0],hover=[-1,-1],hits=[];
let center=[W/2,H*.38,D/2];const mode=document.getElementById('mode');if(!scene.mesh)mode.options[1].disabled=true;
document.getElementById('stats').textContent=`${W} × ${D} × ${H} blocks · ${scene.boxes.filter(b=>b[7]==='object').length} objects`;
function resize(){let d=window.devicePixelRatio||1;canvas.width=meshCanvas.width=innerWidth*d;canvas.height=meshCanvas.height=innerHeight*d;ctx.setTransform(d,0,0,d,0,0);draw()}window.addEventListener('resize',resize);
function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],c=Math.cos(yaw),s=Math.sin(yaw),a=x*c-z*s,b=x*s+z*c;
 let q=b*Math.cos(pitch)-y*Math.sin(pitch),v=b*Math.sin(pitch)+y*Math.cos(pitch),depth=zoom-q;
 let f=Math.min(innerWidth,innerHeight)*.9/Math.max(1,depth);return [innerWidth/2+a*f,innerHeight/2-v*f,depth]}
const faces=[[[0,1,2,3],.84],[[4,5,6,7],1],[[0,1,5,4],.67],[[2,3,7,6],.78],[[1,2,6,5],.9],[[0,3,7,4],.72]];
function shade(hex,k){let n=parseInt(hex.slice(1),16);return `rgb(${Math.round(((n>>16)&255)*k)},${Math.round(((n>>8)&255)*k)},${Math.round((n&255)*k)})`}
function draw(){if(mode.value==='mesh'&&scene.mesh){drawMesh();tip.style.display='none';return}ctx.clearRect(0,0,innerWidth,innerHeight);ctx.fillStyle='#111820';ctx.fillRect(0,0,innerWidth,innerHeight);
 let polys=[],showWalls=document.getElementById('walls').checked,showObjects=document.getElementById('objects').checked,alpha=+document.getElementById('alpha').value/100;hits=[];
 for(let box of scene.boxes){let [x,y,z,w,h,d,col,kind,name]=box;if(kind==='wall'&&!showWalls||kind==='object'&&!showObjects)continue;
 let verts=[[x,y,z],[x+w,y,z],[x+w,y,z+d],[x,y,z+d],[x,y+h,z],[x+w,y+h,z],[x+w,y+h,z+d],[x,y+h,z+d]].map(project);
 for(let [indices,light] of faces){let pp=indices.map(i=>verts[i]);polys.push({pp,depth:pp.reduce((a,p)=>a+p[2],0)/4,color:shade(col,light),alpha:kind==='wall'?alpha:kind==='opening'?.55:1,kind,name})}
 if(kind==='object'){let c=project([x+w/2,y+h/2,z+d/2]);hits.push([c[0],c[1],name])}}
 polys.sort((a,b)=>b.depth-a.depth);for(let f of polys){ctx.globalAlpha=f.alpha;ctx.beginPath();ctx.moveTo(f.pp[0][0],f.pp[0][1]);for(let p of f.pp.slice(1))ctx.lineTo(p[0],p[1]);ctx.closePath();ctx.fillStyle=f.color;ctx.fill();ctx.strokeStyle='#08101655';ctx.lineWidth=.35;ctx.stroke()}ctx.globalAlpha=1;
 if(hover[0]>=0){let best=null,dist=30;for(let p of hits){let d=Math.hypot(p[0]-hover[0],p[1]-hover[1]);if(d<dist){dist=d;best=p}}tip.style.display=best?'block':'none';if(best){tip.textContent=best[2];tip.style.left=(hover[0]+14)+'px';tip.style.top=(hover[1]+14)+'px'}}}
for(let surface of [canvas,meshCanvas]){surface.addEventListener('pointerdown',e=>{drag=true;last=[e.clientX,e.clientY];surface.setPointerCapture(e.pointerId)});
surface.addEventListener('pointerup',()=>drag=false);surface.addEventListener('pointermove',e=>{hover=[e.clientX,e.clientY];if(drag){yaw+=(e.clientX-last[0])*.008;pitch+=(e.clientY-last[1])*.008;last=[e.clientX,e.clientY]}draw()});
surface.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(2,Math.min(100,zoom*Math.exp(e.deltaY*.001)));draw()},{passive:false})}
for(let id of ['walls','objects','alpha'])document.getElementById(id).addEventListener('input',draw);
mode.addEventListener('change',()=>{meshCanvas.style.display=mode.value==='mesh'?'block':'none';canvas.style.display=mode.value==='mesh'?'none':'block';if(mode.value==='mesh'&&scene.mesh){let [a,b]=scene.mesh.bounds;center=a.map((v,i)=>(v+b[i])/2);zoom=Math.max(...b.map((v,i)=>v-a[i]))*1.7}
 else{center=[W/2,H*.38,D/2];zoom=Math.max(W,D,H)*1.7}draw()});
document.getElementById('reset').onclick=()=>{yaw=-.65;pitch=.55;mode.dispatchEvent(new Event('change'))};initMesh();resize();if(gl){mode.value='mesh';mode.dispatchEvent(new Event('change'))}
</script></body></html>'''


def write_viewer(plan, out, mesh_path=None, photo_view=False):
    scene = scene_from_plan(plan)
    scene["mesh"] = mesh_preview_from_ply(mesh_path)
    scene["photo_view"] = photo_view
    encoded = json.dumps(scene, separators=(",", ":")).replace("<", "\\u003c")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__SCENE__", encoded), encoding="utf-8")
    return scene


def write_obj(scene, out):
    """Export the same block geometry for Blender and other mesh tools."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Everything Cubed room", f"mtllib {out.with_suffix('.mtl').name}"]
    materials = {}
    vertex = 1
    for i, (x, y, z, w, h, d, col, kind, name) in enumerate(scene["boxes"]):
        mat = "c" + col.lstrip("#")
        materials[mat] = col
        lines.extend([f"o {kind}_{i}_{''.join(c if c.isalnum() else '_' for c in name)}", f"usemtl {mat}"])
        verts = [(x,y,z),(x+w,y,z),(x+w,y,z+d),(x,y,z+d),
                 (x,y+h,z),(x+w,y+h,z),(x+w,y+h,z+d),(x,y+h,z+d)]
        lines.extend("v %.5f %.5f %.5f" % p for p in verts)
        for face in ((1,4,3,2),(5,6,7,8),(1,2,6,5),(3,4,8,7),(2,3,7,6),(4,1,5,8)):
            lines.append("f " + " ".join(str(vertex+j-1) for j in face))
        vertex += 8
    out.write_text("\n".join(lines)+"\n", encoding="utf-8")
    mtl = []
    for mat, col in materials.items():
        rgb = [int(col[i:i+2], 16)/255 for i in (1,3,5)]
        mtl.extend([f"newmtl {mat}", "Kd %.5f %.5f %.5f" % tuple(rgb), "d 1.0", ""])
    out.with_suffix(".mtl").write_text("\n".join(mtl), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--obj", type=Path)
    opts = parser.parse_args(argv)
    plan = json.loads(opts.plan.read_text())
    html = opts.html or opts.plan.with_name("view.html")
    obj = opts.obj or opts.plan.with_name("model.obj")
    scene = write_viewer(plan, html)
    write_obj(scene, obj)
    print(f"HTML: {html}\nOBJ: {obj}\nMTL: {obj.with_suffix('.mtl')}")


if __name__ == "__main__":
    main()

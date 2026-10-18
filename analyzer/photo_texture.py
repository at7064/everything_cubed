"""Recover per-frame image coordinates from a reconstructed PLY triangle lattice."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from mesh_export import FACE, VERTEX


def read_ply(path):
    path = Path(path)
    with path.open("rb") as stream:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = stream.readline()
            if not line or len(header) > 4096:
                raise ValueError("invalid binary PLY header")
            header += line
    text = header.decode("ascii")
    if "format binary_little_endian 1.0" not in text:
        raise ValueError("photo texture needs a binary little-endian reconstruction PLY")
    nv = int(next(s.split()[2] for s in text.splitlines() if s.startswith("element vertex ")))
    nf = int(next(s.split()[2] for s in text.splitlines() if s.startswith("element face ")))
    vertices = np.memmap(path, dtype=VERTEX, mode="r", offset=len(header), shape=(nv,))
    faces = np.memmap(path, dtype=FACE, mode="r", offset=len(header) + nv * VERTEX.itemsize, shape=(nf,))
    return vertices, faces


def frame_face_ranges(faces):
    """Each view writes one forward-diagonal band then one backward-diagonal band."""
    forward = faces["a"] < faces["b"]
    changes = np.flatnonzero(forward[1:] != forward[:-1]) + 1
    edges = np.r_[0, changes, len(faces)]
    if not forward[0] or len(edges) % 2 != 1 or len(edges) < 3:
        raise ValueError("PLY does not contain ordered image lattices")
    if any(not forward[edges[i]] or forward[edges[i + 1]] for i in range(0, len(edges) - 1, 2)):
        raise ValueError("PLY triangle bands are not ordered by view")
    return [(int(edges[i]), int(edges[i + 2])) for i in range(0, len(edges) - 1, 2)]


def recover_uv(triangles, count):
    """Solve the grid-edge constraints of the triangle topology (Numba keeps large scans fast)."""
    from numba import njit

    @njit(cache=True)
    def solve(tri, n):
        degree = np.zeros(n, np.int32)
        for i in range(len(tri)):
            a, b, c = tri[i, 0], tri[i, 1], tri[i, 2]
            degree[a] += 2
            degree[b] += 1
            degree[c] += 1
        offsets = np.empty(n + 1, np.int32)
        offsets[0] = 0
        for i in range(n):
            offsets[i + 1] = offsets[i] + degree[i]
        cursor = offsets[:-1].copy()
        dest = np.empty(int(offsets[-1]), np.int32)
        dx = np.empty(len(dest), np.int8)
        dy = np.empty(len(dest), np.int8)
        for i in range(len(tri)):
            a, b, c = tri[i, 0], tri[i, 1], tri[i, 2]
            sign = 1 if a < b else -1
            j = cursor[a]; dest[j] = b; dx[j] = sign; dy[j] = 0; cursor[a] += 1
            j = cursor[b]; dest[j] = a; dx[j] = -sign; dy[j] = 0; cursor[b] += 1
            j = cursor[a]; dest[j] = c; dx[j] = 0; dy[j] = sign; cursor[a] += 1
            j = cursor[c]; dest[j] = a; dx[j] = 0; dy[j] = -sign; cursor[c] += 1
        uv = np.zeros((n, 2), np.int16)
        component = np.full(n, -1, np.int32)
        queue = np.empty(n, np.int32)
        sizes = np.empty(n, np.int32)
        nc = 0
        for root in range(n):
            if component[root] != -1 or degree[root] == 0:
                continue
            head, tail = 0, 1
            queue[0] = root
            component[root] = nc
            while head < tail:
                current = queue[head]; head += 1
                for j in range(offsets[current], offsets[current + 1]):
                    neighbor = dest[j]
                    if component[neighbor] == -1:
                        component[neighbor] = nc
                        uv[neighbor, 0] = uv[current, 0] + dx[j]
                        uv[neighbor, 1] = uv[current, 1] + dy[j]
                        queue[tail] = neighbor; tail += 1
            sizes[nc] = tail
            nc += 1
        best = np.argmax(sizes[:nc]) if nc else -1
        return uv, component == best

    return solve(np.ascontiguousarray(triangles, dtype=np.int32), count)


def crop_frame(path, aspect):
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    width, height = image.size
    if width / height > aspect:
        new_width = round(height * aspect)
        left = (width - new_width) // 2
        image = image.crop((left, 0, left + new_width, height))
    elif width / height < aspect:
        new_height = round(width / aspect)
        top = (height - new_height) // 2
        image = image.crop((0, top, width, top + new_height))
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=90, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def photo_patches(vertices, vertex_start, uv, connected, uv_min, size, tri, spacing):
    """Cut a frame into continuous image patches; split them where depth has no mesh edge."""
    width, height = map(int, size)
    lookup = np.full((height, width), -1, np.int32)
    ids = np.flatnonzero(connected)
    xy = uv[ids] - uv_min
    lookup[xy[:, 1], xy[:, 0]] = ids
    # A missing fine-mesh edge marks an occlusion or depth break, not a surface to stretch a photo across.
    horizontal = np.zeros((height, width - 1), bool)
    vertical = np.zeros((height - 1, width), bool)
    for col_a, col_b in ((0, 1), (0, 2)):
        a, b = tri[:, col_a], tri[:, col_b]
        keep = connected[a] & connected[b]
        p, q = uv[a[keep]] - uv_min, uv[b[keep]] - uv_min
        h = np.abs(p[:, 0] - q[:, 0]) == 1
        v = np.abs(p[:, 1] - q[:, 1]) == 1
        horizontal[p[h, 1], np.minimum(p[h, 0], q[h, 0])] = True
        vertical[np.minimum(p[v, 1], q[v, 1]), p[v, 0]] = True
    xs = np.unique(np.r_[np.arange(0, width, spacing), width - 1])
    ys = np.unique(np.r_[np.arange(0, height, spacing), height - 1])
    records = []
    for y0, y1 in zip(ys[:-1], ys[1:]):
        for x0, x1 in zip(xs[:-1], xs[1:]):
            corners = [lookup[y0, x0], lookup[y0, x1], lookup[y1, x0], lookup[y1, x1]]
            if min(corners) < 0:
                continue
            continuity = (horizontal[y0:y1 + 1, x0:x1].sum() + vertical[y0:y1, x0:x1 + 1].sum())
            possible = (y1 - y0 + 1) * (x1 - x0) + (y1 - y0) * (x1 - x0 + 1)
            if continuity < 0.985 * possible:
                continue
            for a, b, c in ((0, 1, 2), (3, 2, 1)):
                for index in (a, b, c):
                    vertex = vertices[vertex_start + corners[index]]
                    x, y = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))[index]
                    records.append((vertex["x"], vertex["y"], vertex["z"],
                                    (x + 0.5) / width, (y + 0.5) / height))
    return np.asarray(records, dtype="<f4").reshape(-1, 5)


def photo_scene(ply, frames, max_faces_per_frame=6000):
    """Return compact textured triangle batches, one original photo for each view."""
    vertices, faces = read_ply(ply)
    ranges = frame_face_ranges(faces)
    frames = list(frames)
    if len(ranges) != len(frames):
        raise ValueError(f"mesh has {len(ranges)} views but {len(frames)} photo frames were supplied")
    batches = []
    bounds_lo = np.full(3, np.inf)
    bounds_hi = np.full(3, -np.inf)
    record = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("u", "<f4"), ("v", "<f4")])
    for index, ((start, stop), path) in enumerate(zip(ranges, frames)):
        block = faces[start:stop]
        vertex_start = min(int(block[k].min()) for k in ("a", "b", "c"))
        vertex_stop = max(int(block[k].max()) for k in ("a", "b", "c")) + 1
        tri = np.stack([block[k] - vertex_start for k in ("a", "b", "c")], axis=1)
        uv, connected = recover_uv(tri, vertex_stop - vertex_start)
        good_uv = uv[connected]
        uv_min = good_uv.min(axis=0)
        uv_max = good_uv.max(axis=0)
        size = (uv_max - uv_min + 1).astype(np.int32)
        spacing = max(2, int(np.sqrt(2 * int(size[0]) * int(size[1]) / max_faces_per_frame)))
        main = photo_patches(vertices, vertex_start, uv, connected, uv_min, size, tri, spacing)
        if not len(main):
            step = max(1, int(np.ceil(len(tri) / max_faces_per_frame)))
            sample = tri[::step]
            sample = sample[connected[sample].all(axis=1)]
            ids = sample.ravel()
            source = vertices[vertex_start + ids]
            main = np.stack([source["x"], source["y"], source["z"],
                             (uv[ids, 0] - uv_min[0] + 0.5) / size[0],
                             (uv[ids, 1] - uv_min[1] + 0.5) / size[1]], axis=1).astype("<f4")
        if not len(main):
            raise ValueError(f"no textured geometry could be recovered for {path}")
        packed = main.view(record).reshape(-1)
        for axis, key in enumerate(("x", "y", "z")):
            bounds_lo[axis] = min(bounds_lo[axis], float(packed[key].min()))
            bounds_hi[axis] = max(bounds_hi[axis], float(packed[key].max()))
        fallback = photo_patches(vertices, vertex_start, uv, connected, uv_min, size, tri, 28)
        if not len(fallback):
            triangles = main.reshape(-1, 3, 5)
            fallback = triangles[:: max(1, len(triangles) // 1000)].reshape(-1, 5)
        batches.append({"name": Path(path).name, "grid": size.tolist(), "faces": len(packed) // 3,
                        "image": crop_frame(path, float(size[0] / size[1])),
                        "coarse_b64": base64.b64encode(fallback.tobytes()).decode("ascii"),
                        "vertices_b64": base64.b64encode(packed.tobytes()).decode("ascii")})
        print(f"[photo] frame {index + 1}/{len(frames)}: {len(packed) // 3} photo-patch triangles, grid {size[0]}x{size[1]}", flush=True)
    return {"bounds": [bounds_lo.tolist(), bounds_hi.tolist()], "batches": batches,
            "total_faces": sum(b["faces"] for b in batches)}


PHOTO_HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Everything Cubed · photo-textured 3D</title><style>
*{box-sizing:border-box}body{margin:0;background:#111820;color:#edf2f2;font:14px system-ui,sans-serif;overflow:hidden}
canvas{display:block;width:100vw;height:100vh;cursor:grab}canvas:active{cursor:grabbing}
.panel{position:fixed;left:18px;top:18px;background:#17232ce8;border:1px solid #39505a;border-radius:12px;padding:16px;max-width:310px;box-shadow:0 8px 30px #0007}
h1{font-size:17px;margin:0 0 7px}.muted{color:#acc0c7;font-size:12px;line-height:1.5}
label{display:block;margin-top:12px}input[type=range]{width:100%}button,a{margin-top:9px;border:1px solid #5a7480;background:#243945;color:#eff;border-radius:7px;padding:7px 10px;cursor:pointer;text-decoration:none;display:inline-block}
</style></head><body><canvas id="view"></canvas><div class="panel"><h1>Photo-textured 3D</h1>
<div id="stats" class="muted"></div><div class="muted">Original video pixels cut into patches and placed at their measured 3D positions. Drag to orbit · scroll to zoom.</div>
<label><input id="combine" type="checkbox" checked> <span id="combineLabel">Combine all frames</span></label>
<label>Frame <span id="frameName"></span><input id="frame" type="range" min="0" max="0" value="0"></label>
<button id="reset">Reset view</button> <a href="view.html">Colored solid / room model</a></div><script>
const scene=__SCENE__,canvas=document.getElementById('view'),gl=canvas.getContext('webgl',{antialias:true}),fallbackCtx=gl?null:canvas.getContext('2d');
const frameSlider=document.getElementById('frame'),combine=document.getElementById('combine'),frameName=document.getElementById('frameName');
frameSlider.max=scene.batches.length-1;document.getElementById('stats').textContent=`${scene.batches.length} original frames · ${scene.total_faces.toLocaleString()} textured triangles`;
const [lo,hi]=scene.bounds;let center=lo.map((v,i)=>(v+hi[i])/2),span=Math.max(...lo.map((v,i)=>hi[i]-v));
let yaw=-.65,pitch=.55,zoom=span*1.1,drag=false,last=[0,0],program=null,batches=[];
function shader(type,source){let s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}
function setup(){if(!gl){document.getElementById('stats').textContent=`${scene.batches.length} original frames · CPU photo preview`;
 document.getElementById('combineLabel').textContent='Combine 8 keyframes (CPU preview)';
 for(let item of scene.batches){let raw=atob(item.coarse_b64),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
  let image=new Image(),entry={array:new Float32Array(bytes.buffer),count:bytes.length/20,image};image.onload=()=>draw();image.src=item.image;batches.push(entry)}
 resize();return}
 const vs=shader(gl.VERTEX_SHADER,`attribute vec3 aPos;attribute vec2 aUv;varying vec2 vUv;uniform vec3 uCenter;uniform float uYaw,uPitch,uZoom,uAspect;
 void main(){vec3 p=aPos-uCenter;float c=cos(uYaw),s=sin(uYaw);float a=p.x*c-p.z*s,b=p.x*s+p.z*c;float q=b*cos(uPitch)-p.y*sin(uPitch),v=b*sin(uPitch)+p.y*cos(uPitch);float depth=max(.1,uZoom-q);gl_Position=vec4(a/uAspect,v,depth*1.0002-.2,depth);vUv=aUv;}`);
 const fs=shader(gl.FRAGMENT_SHADER,`precision mediump float;varying vec2 vUv;uniform sampler2D uImage;void main(){gl_FragColor=texture2D(uImage,vUv);}`);
 program=gl.createProgram();gl.attachShader(program,vs);gl.attachShader(program,fs);gl.linkProgram(program);
 if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
 gl.uniform1i(gl.getUniformLocation(program,'uImage'),0);gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);
 for(let item of scene.batches){let raw=atob(item.vertices_b64),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
  let buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,bytes,gl.STATIC_DRAW);
  let image=new Image(),entry={buffer,count:bytes.length/20,image,texture:null};image.onload=()=>{draw()};image.src=item.image;batches.push(entry)}
 resize()}
function upload(entry){if(entry.texture||!entry.image.complete||!entry.image.naturalWidth)return;
 entry.texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,entry.texture);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);
 gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,entry.image);
 gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
 gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE)}
function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],c=Math.cos(yaw),s=Math.sin(yaw),a=x*c-z*s,b=x*s+z*c;
 let q=b*Math.cos(pitch)-y*Math.sin(pitch),v=b*Math.sin(pitch)+y*Math.cos(pitch),depth=zoom-q;
 let f=Math.min(innerWidth,innerHeight)*.9/Math.max(.1,depth);return [innerWidth/2+a*f,innerHeight/2-v*f,depth]}
function drawFallback(){let ctx=fallbackCtx;ctx.setTransform(devicePixelRatio||1,0,0,devicePixelRatio||1,0,0);
 ctx.fillStyle='#111820';ctx.fillRect(0,0,innerWidth,innerHeight);let polygons=[];
 let list=combine.checked?batches.filter((_,i)=>i%Math.max(1,Math.ceil(batches.length/8))===0):[batches[+frameSlider.value]];
 for(let entry of list){if(!entry||!entry.image.complete||!entry.image.naturalWidth)continue;
  let n=entry.count/3,limit=n,data=entry.array;
  for(let i=0;i<limit;i++){let offset=Math.floor(i*n/limit)*15,points=[],source=[];
   for(let j=0;j<3;j++){let k=offset+j*5;points.push(project([data[k],data[k+1],data[k+2]]));source.push([data[k+3]*entry.image.width,data[k+4]*entry.image.height])}
   polygons.push({points,source,image:entry.image,depth:(points[0][2]+points[1][2]+points[2][2])/3})}}
 polygons.sort((a,b)=>b.depth-a.depth);
 for(let face of polygons){let p=face.points,s=face.source,den=(s[1][0]-s[0][0])*(s[2][1]-s[0][1])-(s[2][0]-s[0][0])*(s[1][1]-s[0][1]);if(Math.abs(den)<.01)continue;
  let a=((p[1][0]-p[0][0])*(s[2][1]-s[0][1])-(p[2][0]-p[0][0])*(s[1][1]-s[0][1]))/den;
  let b=((p[1][1]-p[0][1])*(s[2][1]-s[0][1])-(p[2][1]-p[0][1])*(s[1][1]-s[0][1]))/den;
  let c=((p[2][0]-p[0][0])*(s[1][0]-s[0][0])-(p[1][0]-p[0][0])*(s[2][0]-s[0][0]))/den;
  let d=((p[2][1]-p[0][1])*(s[1][0]-s[0][0])-(p[1][1]-p[0][1])*(s[2][0]-s[0][0]))/den;
  let e=p[0][0]-a*s[0][0]-c*s[0][1],f=p[0][1]-b*s[0][0]-d*s[0][1];
  ctx.save();ctx.beginPath();ctx.moveTo(p[0][0],p[0][1]);ctx.lineTo(p[1][0],p[1][1]);ctx.lineTo(p[2][0],p[2][1]);ctx.closePath();ctx.clip();
  ctx.transform(a,b,c,d,e,f);ctx.drawImage(face.image,0,0);ctx.restore()}
 frameName.textContent=scene.batches[+frameSlider.value].name;frameSlider.disabled=combine.checked}
function draw(){if(!gl){drawFallback();return}if(!program)return;gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(.067,.094,.125,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
 gl.uniform3fv(gl.getUniformLocation(program,'uCenter'),center);gl.uniform1f(gl.getUniformLocation(program,'uYaw'),yaw);
 gl.uniform1f(gl.getUniformLocation(program,'uPitch'),pitch);gl.uniform1f(gl.getUniformLocation(program,'uZoom'),zoom);
 gl.uniform1f(gl.getUniformLocation(program,'uAspect'),innerWidth/innerHeight);
 let list=combine.checked?batches:[batches[+frameSlider.value]];for(let entry of list){if(!entry)continue;upload(entry);if(!entry.texture)continue;
  gl.bindBuffer(gl.ARRAY_BUFFER,entry.buffer);let pos=gl.getAttribLocation(program,'aPos'),uv=gl.getAttribLocation(program,'aUv');
  gl.enableVertexAttribArray(pos);gl.vertexAttribPointer(pos,3,gl.FLOAT,false,20,0);gl.enableVertexAttribArray(uv);gl.vertexAttribPointer(uv,2,gl.FLOAT,false,20,12);
  gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,entry.texture);gl.drawArrays(gl.TRIANGLES,0,entry.count)}
 frameName.textContent=scene.batches[+frameSlider.value].name;frameSlider.disabled=combine.checked}
function resize(){let d=window.devicePixelRatio||1;canvas.width=innerWidth*d;canvas.height=innerHeight*d;draw()}window.addEventListener('resize',resize);
canvas.addEventListener('pointerdown',e=>{drag=true;last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointerup',()=>drag=false);canvas.addEventListener('pointermove',e=>{if(drag){yaw+=(e.clientX-last[0])*.008;pitch+=(e.clientY-last[1])*.008;last=[e.clientX,e.clientY];draw()}});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.5,Math.min(1000,zoom*Math.exp(e.deltaY*.001)));draw()},{passive:false});
frameSlider.addEventListener('input',draw);combine.addEventListener('change',draw);document.getElementById('reset').onclick=()=>{yaw=-.65;pitch=.55;zoom=span*1.1;draw()};setup();
</script></body></html>'''


def write_photo_viewer(ply, frames, out, max_faces_per_frame=6000):
    scene = photo_scene(ply, frames, max_faces_per_frame=max_faces_per_frame)
    encoded = json.dumps(scene, separators=(",", ":")).replace("<", "\\u003c")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PHOTO_HTML.replace("__SCENE__", encoded), encoding="utf-8")
    return scene

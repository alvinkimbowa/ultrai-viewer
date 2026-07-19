"use strict";

const DEFAULT_LOCATIONS = ["wrist", "mid-arm", "elbow"];
const DEFAULT_NERVES = ["ulnar", "median", "radial", "plex", "lfcn", "peroneal", "fibular", "tibial", "sural", "proximal", "accessory", "quad", "sciatic", "unknown"];
const DEFAULT_ANATOMY = ["muscle", "artery", "vein", "skin", "subcutaneous tissue", "cartilage", "tendon", "bone"];
const HANDLE_DATABASE = "ultrai-annotator";
const TOOL_SETTINGS_KEY = "ultrai-tool-settings";
const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const video = $("video");

const state = {
  media: [], mediaIndex: -1, outputDir: null, activeClass: null, location: null,
  manifestVideos: [],
  locations: [...DEFAULT_LOCATIONS], nerves: [...DEFAULT_NERVES], anatomy: [...DEFAULT_ANATOMY],
  instancesByKey: new Map(), undoByKey: new Map(), redoByKey: new Map(), nextIdsByKey: new Map(),
  sourceWidth: 0, sourceHeight: 0, frameIndex: 0, frameCount: 1, fps: 30,
  scale: 1, offsetX: 0, offsetY: 0, fit: true, drawing: false, moving: false,
  points: [], moveIndex: -1, moveAnchor: null, moveOriginal: null, lastPointer: null,
};
let autoSavePromise=Promise.resolve();

function status(text) { $("status").textContent = text; }
function saveToolSettings(){
  const settings={tool:$("tool").value,showMasks:$("showMasks").checked,fillMasks:$("fillMasks").checked,opacity:$("opacity").value,radius:$("radius").value};localStorage.setItem(TOOL_SETTINGS_KEY,JSON.stringify(settings));
}
function restoreToolSettings(){
  try{const settings=JSON.parse(localStorage.getItem(TOOL_SETTINGS_KEY)||"null");if(!settings)return;if(["select","freehand","polygon","eraser"].includes(settings.tool))$("tool").value=settings.tool;if(typeof settings.showMasks==="boolean")$("showMasks").checked=settings.showMasks;if(typeof settings.fillMasks==="boolean")$("fillMasks").checked=settings.fillMasks;const opacity=Number(settings.opacity),radius=Number(settings.radius);if(opacity>=0&&opacity<=100)$("opacity").value=String(opacity);if(radius>=1&&radius<=50)$("radius").value=String(radius);}catch{}
}
function cleanClass(value) { return String(value || "").trim().toLowerCase(); }
function safeClass(value) { return cleanClass(value).replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "unlabeled"; }
function mediaItem() { return state.media[state.mediaIndex] || null; }
function annotationKey() {
  const item = mediaItem();
  if (!item) return "";
  return item.type === "video" ? `${item.id}:frame:${state.frameIndex}` : item.id;
}
function instances() {
  const key = annotationKey();
  if (!state.instancesByKey.has(key)) state.instancesByKey.set(key, []);
  return state.instancesByKey.get(key);
}
function classOrder() { return [...state.nerves, ...state.anatomy]; }
function classColor(name) {
  let index = classOrder().indexOf(cleanClass(name));
  if (index < 0) index = classOrder().length;
  const hue = (index * 137.507764) % 360;
  return `hsl(${hue} 78% 55%)`;
}
function parseHslColor(name) {
  const tmp = document.createElement("canvas").getContext("2d");
  tmp.fillStyle = classColor(name); tmp.fillRect(0, 0, 1, 1);
  return tmp.getImageData(0, 0, 1, 1).data;
}

function cloneInstances(list) { return list.map((x) => ({className:x.className, id:x.id, mask:new Uint8Array(x.mask)})); }
function pushHistory() {
  const key = annotationKey(); if (!key) return;
  const stack = state.undoByKey.get(key) || [];
  stack.push(cloneInstances(instances()));
  if (stack.length > 80) stack.shift();
  state.undoByKey.set(key, stack); state.redoByKey.set(key, []);
}
function resetHistory() { const key=annotationKey(); if(key){state.undoByKey.set(key,[cloneInstances(instances())]);state.redoByKey.set(key,[]);} }
function undo() {
  const key=annotationKey(), stack=state.undoByKey.get(key)||[]; if(stack.length<=1)return;
  const current=stack.pop(); const redo=state.redoByKey.get(key)||[]; redo.push(current); state.redoByKey.set(key,redo);
  state.instancesByKey.set(key,cloneInstances(stack[stack.length-1])); render();queueAutoSave();
}
function redo() {
  const key=annotationKey(), redoStack=state.redoByKey.get(key)||[]; if(!redoStack.length)return;
  const restored=redoStack.pop(); state.redoByKey.set(key,redoStack); state.instancesByKey.set(key,cloneInstances(restored));
  (state.undoByKey.get(key)||[]).push(cloneInstances(restored)); render();queueAutoSave();
}

function rebuildChips(containerId, labels, kind) {
  const holder=$(containerId),addButton=holder.querySelector("[data-add-label]"); holder.replaceChildren();
  for(const label of labels){
    const button=document.createElement("button"); button.className="chip"; button.textContent=label==="lfcn"?"LFCN":label.replace(/\b\w/g,c=>c.toUpperCase());
    button.dataset.value=label;
    button.addEventListener("click",()=>{
      if(kind==="location") { state.location=label; if(mediaItem())mediaItem().location=label; }
      else { state.activeClass=label; }
      updateChipSelection();if(kind==="location")queueAutoSave();
    }); holder.appendChild(button);
  }
  if(addButton)holder.appendChild(addButton);
}
function updateChipSelection(){
  document.querySelectorAll("#locations .chip").forEach(b=>b.classList.toggle("selected",b.dataset.value===state.location));
  document.querySelectorAll("#nerves .chip,#anatomy .chip").forEach(b=>b.classList.toggle("selected",b.dataset.value===state.activeClass));
}
function rebuildAllChips(){rebuildChips("locations",state.locations,"location");rebuildChips("nerves",state.nerves,"class");rebuildChips("anatomy",state.anatomy,"class");updateChipSelection();}
function addLabel(kind){
  const value=cleanClass(prompt(`New ${kind} name:`)); if(!value)return;
  const target=kind==="location"?state.locations:(kind==="nerve"?state.nerves:state.anatomy);
  if([...state.locations,...state.nerves,...state.anatomy].includes(value)){alert("That label already exists.");return;}
  target.push(value); rebuildAllChips();queueAutoSave();
}

function openHandleDatabase(){
  return new Promise((resolve,reject)=>{const request=indexedDB.open(HANDLE_DATABASE,1);request.onupgradeneeded=()=>request.result.createObjectStore("handles");request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);});
}
async function rememberHandle(name,handle){
  try{const database=await openHandleDatabase();await new Promise((resolve,reject)=>{const transaction=database.transaction("handles","readwrite");transaction.objectStore("handles").put(handle,name);transaction.oncomplete=resolve;transaction.onerror=()=>reject(transaction.error);});database.close();}catch{}
}
async function recalledHandle(name){
  try{const database=await openHandleDatabase(),handle=await new Promise((resolve,reject)=>{const request=database.transaction("handles").objectStore("handles").get(name);request.onsuccess=()=>resolve(request.result||null);request.onerror=()=>reject(request.error);});database.close();return handle;}catch{return null;}
}
async function pickFiles(types,kind,multiple=true){
  try{const previous=await recalledHandle(kind),options={id:`ultrai-${kind}`,multiple,types};if(previous)options.startIn=previous;const handles=await window.showOpenFilePicker(options);if(handles.length)await rememberHandle(kind,handles[0]);return handles;}
  catch(error){if(error.name!=="AbortError")alert(error.message);return [];}
}
async function loadImages(){
  const handles=await pickFiles([{description:"Images",accept:{"image/*":[".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff"]}}],"images");
  if(!handles.length)return; await addMedia(handles,"image");
}
async function loadVideos(){
  const handles=await pickFiles([{description:"Videos",accept:{"video/*":[".mp4",".webm",".mov",".m4v"]}}],"videos");
  if(!handles.length)return; await addMedia(handles,"video");
}
async function addMedia(handles,type){
  clearMediaUrls(); state.media=[];
  for(const handle of handles){const file=await handle.getFile();state.media.push({id:`${type}:${file.name}:${file.size}:${file.lastModified}`,name:file.name,type,handle,file,url:URL.createObjectURL(file)});}
  state.mediaIndex=0; rebuildMediaList(); await openMedia(0);
}
function clearMediaUrls(){for(const item of state.media)if(item.url)URL.revokeObjectURL(item.url);}
function rebuildMediaList(){
  const select=$("mediaList"); select.replaceChildren(); state.media.forEach((item,i)=>{const o=document.createElement("option");o.value=i;o.textContent=item.name;select.appendChild(o);});
  select.disabled=!state.media.length; updateNavigation();
}
async function openMedia(index){
  if(index<0||index>=state.media.length)return;await flushAutoSave();pauseVideo(); state.mediaIndex=index; $("mediaList").value=String(index);
  state.activeClass=null; state.location=null; updateChipSelection();
  const item=mediaItem();
  if(item.type==="image"){
    video.removeAttribute("src");
    if(/\.tiff?$/i.test(item.name)){
      const buffer=await item.file.arrayBuffer(),pages=UTIF.decode(buffer);if(!pages.length)throw new Error("TIFF contains no images");UTIF.decodeImage(buffer,pages[0]);const rgba=UTIF.toRGBA8(pages[0]),surface=document.createElement("canvas");surface.width=pages[0].width;surface.height=pages[0].height;surface.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(rgba),surface.width,surface.height),0,0);item.element=surface;state.sourceWidth=surface.width;state.sourceHeight=surface.height;
    }else{const image=new Image();image.src=item.url;await image.decode();item.element=image;state.sourceWidth=image.naturalWidth;state.sourceHeight=image.naturalHeight;}
    state.frameIndex=0;state.frameCount=1;
  } else {
    video.src=item.url; video.load(); await once(video,"loadedmetadata");
    state.sourceWidth=video.videoWidth;state.sourceHeight=video.videoHeight;state.frameIndex=0;state.fps=30;state.frameCount=Math.max(1,Math.floor(video.duration*state.fps));
    video.currentTime=0; await once(video,"seeked").catch(()=>{});
    await loadManifestLocation(item.name);
  }
  state.fit=true; resizeCanvas(); await loadSavedMasksForCurrent(); if(!state.undoByKey.has(annotationKey()))resetHistory(); updateNavigation(); render(); status(`Loaded ${item.name}`);
}
function once(target,event){return new Promise((resolve,reject)=>{const done=()=>{cleanup();resolve();};const fail=()=>{cleanup();reject(new Error(`Failed waiting for ${event}`));};const cleanup=()=>{target.removeEventListener(event,done);target.removeEventListener("error",fail);};target.addEventListener(event,done,{once:true});target.addEventListener("error",fail,{once:true});setTimeout(done,5000);});}
async function setFrame(index){
  const item=mediaItem();if(!item||item.type!=="video")return; index=Math.max(0,Math.min(state.frameCount-1,index));await flushAutoSave();
  const previousIndex=state.frameIndex,sourceGray=index===previousIndex+1?videoGrayFrame():null,sourceInstances=sourceGray?cloneInstances(instances()):[];
  state.frameIndex=index;video.currentTime=Math.min(video.duration||0,index/state.fps);await once(video,"seeked").catch(()=>{});
  const loaded=await loadSavedMasksForCurrent();
  if(!loaded&&sourceGray&&sourceInstances.length&&(!state.instancesByKey.has(annotationKey())||instances().length===0)){
    const targetGray=videoGrayFrame(),propagated=sourceInstances.map(instance=>({...instance,mask:propagateMask(instance.mask,sourceGray,targetGray)}));
    state.instancesByKey.set(annotationKey(),propagated);restoreNextIds(propagated);status(`Propagated ${propagated.length} mask(s) to frame ${index+1}`);queueAutoSave();
  }
  if(!state.undoByKey.has(annotationKey()))resetHistory();updateNavigation();render();
}
function updateNavigation(){
  const has=!!mediaItem(), isVideo=has&&mediaItem().type==="video";
  $("prevMedia").disabled=!has||state.mediaIndex<=0;$("nextMedia").disabled=!has||state.mediaIndex>=state.media.length-1;
  for(const id of ["firstFrame","prevFrame","play","nextFrame","lastFrame","frameSlider"])$(id).disabled=!isVideo;
  $("frameSlider").max=Math.max(0,state.frameCount-1);$("frameSlider").value=state.frameIndex;
  $("frameLabel").textContent=isVideo?`${state.frameIndex+1} / ${state.frameCount}`:"1 / 1";
}
function pauseVideo(){video.pause();$("play").textContent="▶";}
async function togglePlayback(){if(video.paused){await flushAutoSave();video.play();$("play").textContent="❚❚";requestAnimationFrame(playLoop);}else pauseVideo();}
function playLoop(){if(video.paused)return;state.frameIndex=Math.min(state.frameCount-1,Math.floor(video.currentTime*state.fps));updateNavigation();render();requestAnimationFrame(playLoop);}

function resizeCanvas(){const rect=$("stage").getBoundingClientRect();canvas.width=Math.max(1,Math.round(rect.width));canvas.height=Math.max(1,Math.round(rect.height));if(state.fit)fitView();render();}
function fitView(){if(!state.sourceWidth||!state.sourceHeight)return;state.scale=Math.min(canvas.width/state.sourceWidth,canvas.height/state.sourceHeight);state.offsetX=(canvas.width-state.sourceWidth*state.scale)/2;state.offsetY=(canvas.height-state.sourceHeight*state.scale)/2;state.fit=true;}
function toImagePoint(event){const rect=canvas.getBoundingClientRect();const x=(event.clientX-rect.left-state.offsetX)/state.scale,y=(event.clientY-rect.top-state.offsetY)/state.scale;if(x<0||y<0||x>=state.sourceWidth||y>=state.sourceHeight)return null;return{x:Math.floor(x),y:Math.floor(y)};}
function render(){
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle="#252525";ctx.fillRect(0,0,canvas.width,canvas.height);const item=mediaItem();if(!item)return;
  const source=item.type==="video"?video:item.element; if(source)ctx.drawImage(source,state.offsetX,state.offsetY,state.sourceWidth*state.scale,state.sourceHeight*state.scale);
  if($("showMasks").checked)renderInstances();renderWorkingLine();
}
function renderInstances(){
  if(!state.sourceWidth)return;const overlay=document.createElement("canvas");overlay.width=state.sourceWidth;overlay.height=state.sourceHeight;const ox=overlay.getContext("2d");
  const image=ox.createImageData(state.sourceWidth,state.sourceHeight),data=image.data,filled=$("fillMasks").checked,alpha=Math.round(Number($("opacity").value)*2.55);
  for(const instance of instances()){
    const [r,g,b]=parseHslColor(instance.className);
    for(let i=0;i<instance.mask.length;i++){
      if(!instance.mask[i])continue;
      let visible=filled;
      if(!filled){const x=i%state.sourceWidth,y=Math.floor(i/state.sourceWidth);visible=x===0||y===0||x===state.sourceWidth-1||y===state.sourceHeight-1||!instance.mask[i-1]||!instance.mask[i+1]||!instance.mask[i-state.sourceWidth]||!instance.mask[i+state.sourceWidth];}
      if(visible){const p=i*4;data[p]=r;data[p+1]=g;data[p+2]=b;data[p+3]=alpha;}
    }
  }
  ox.putImageData(image,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(overlay,state.offsetX,state.offsetY,state.sourceWidth*state.scale,state.sourceHeight*state.scale);ctx.imageSmoothingEnabled=true;
}
function renderWorkingLine(){if(state.points.length<1)return;ctx.strokeStyle=state.activeClass?classColor(state.activeClass):"#00ff00";ctx.lineWidth=2;ctx.beginPath();state.points.forEach((p,i)=>{const x=state.offsetX+p.x*state.scale,y=state.offsetY+p.y*state.scale;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}

function maskFromPolygon(points){
  const temp=document.createElement("canvas");temp.width=state.sourceWidth;temp.height=state.sourceHeight;const tc=temp.getContext("2d");tc.fillStyle="white";tc.beginPath();points.forEach((p,i)=>i?tc.lineTo(p.x,p.y):tc.moveTo(p.x,p.y));tc.closePath();tc.fill();
  const pixels=tc.getImageData(0,0,temp.width,temp.height).data,mask=new Uint8Array(temp.width*temp.height);for(let i=0;i<mask.length;i++)mask[i]=pixels[i*4+3]?1:0;return mask;
}
function completePolygon(){
  if(state.points.length<3)return cancelDrawing();if(!state.activeClass){alert("Select a nerve or other anatomy before drawing.");return cancelDrawing();}
  const key=annotationKey(), idMap=state.nextIdsByKey.get(key)||new Map(),id=idMap.get(state.activeClass)||1;idMap.set(state.activeClass,id+1);state.nextIdsByKey.set(key,idMap);
  instances().push({className:state.activeClass,id,mask:maskFromPolygon(state.points)});state.points=[];state.drawing=false;pushHistory();render();status(`Created ${state.activeClass} ${id}`);queueAutoSave();
}
function cancelDrawing(){state.points=[];state.drawing=false;render();}
function hitInstance(point){for(let i=instances().length-1;i>=0;i--){if(instances()[i].mask[point.y*state.sourceWidth+point.x])return i;}return-1;}
function translateMask(mask,dx,dy){const moved=new Uint8Array(mask.length);for(let y=0;y<state.sourceHeight;y++)for(let x=0;x<state.sourceWidth;x++){if(!mask[y*state.sourceWidth+x])continue;const nx=x+dx,ny=y+dy;if(nx>=0&&ny>=0&&nx<state.sourceWidth&&ny<state.sourceHeight)moved[ny*state.sourceWidth+nx]=1;}return moved;}
function videoGrayFrame(){
  const frame=document.createElement("canvas");frame.width=state.sourceWidth;frame.height=state.sourceHeight;const fc=frame.getContext("2d",{willReadFrequently:true});fc.drawImage(video,0,0,frame.width,frame.height);const rgba=fc.getImageData(0,0,frame.width,frame.height).data,gray=new Uint8Array(frame.width*frame.height);for(let i=0;i<gray.length;i++)gray[i]=(rgba[i*4]*77+rgba[i*4+1]*150+rgba[i*4+2]*29)>>8;return gray;
}
function propagateMask(mask,source,target){
  let minX=state.sourceWidth,minY=state.sourceHeight,maxX=-1,maxY=-1;for(let i=0;i<mask.length;i++)if(mask[i]){const x=i%state.sourceWidth,y=Math.floor(i/state.sourceWidth);minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);}if(maxX<0)return new Uint8Array(mask);
  const step=Math.max(1,Math.floor(Math.max(maxX-minX,maxY-minY)/40)),radius=12;let bestDx=0,bestDy=0,bestScore=Infinity;
  for(let dy=-radius;dy<=radius;dy++)for(let dx=-radius;dx<=radius;dx++){let score=0,count=0;for(let y=minY;y<=maxY;y+=step)for(let x=minX;x<=maxX;x+=step){const i=y*state.sourceWidth+x;if(!mask[i])continue;const nx=x+dx,ny=y+dy;if(nx<0||ny<0||nx>=state.sourceWidth||ny>=state.sourceHeight)continue;score+=Math.abs(source[i]-target[ny*state.sourceWidth+nx]);count++;}if(count&&score/count<bestScore){bestScore=score/count;bestDx=dx;bestDy=dy;}}
  return translateMask(mask,bestDx,bestDy);
}
function restoreNextIds(list){const ids=new Map();for(const instance of list)ids.set(instance.className,Math.max(ids.get(instance.className)||1,instance.id+1));state.nextIdsByKey.set(annotationKey(),ids);}
function eraseLine(a,b){
  const radius=Number($("radius").value),steps=Math.max(1,Math.ceil(Math.hypot(b.x-a.x,b.y-a.y)));
  for(let step=0;step<=steps;step++){const cx=Math.round(a.x+(b.x-a.x)*step/steps),cy=Math.round(a.y+(b.y-a.y)*step/steps);for(let y=Math.max(0,cy-radius);y<=Math.min(state.sourceHeight-1,cy+radius);y++)for(let x=Math.max(0,cx-radius);x<=Math.min(state.sourceWidth-1,cx+radius);x++)if((x-cx)**2+(y-cy)**2<=radius**2)for(const item of instances())item.mask[y*state.sourceWidth+x]=0;}
}
canvas.addEventListener("pointerdown",(event)=>{
  const p=toImagePoint(event);if(!p)return;canvas.setPointerCapture(event.pointerId);const tool=$("tool").value;
  const index=tool!=="eraser"&&!(tool==="polygon"&&state.drawing)?hitInstance(p):-1;
  if(index>=0){state.moving=true;state.moveIndex=index;state.moveAnchor=p;state.moveOriginal=new Uint8Array(instances()[index].mask);}
  else if(tool==="select")return;
  else if(tool==="freehand"){if(!state.activeClass){alert("Select a nerve or other anatomy first.");return;}state.drawing=true;state.points=[p];}
  else if(tool==="polygon"){if(!state.activeClass){alert("Select a nerve or other anatomy first.");return;}state.points.push(p);state.drawing=true;render();}
  else if(tool==="eraser"){state.drawing=true;state.lastPointer=p;eraseLine(p,p);render();}
});
canvas.addEventListener("pointermove",(event)=>{
  const p=toImagePoint(event);if(!p)return;const tool=$("tool").value;
  if(state.moving){instances()[state.moveIndex].mask=translateMask(state.moveOriginal,p.x-state.moveAnchor.x,p.y-state.moveAnchor.y);render();}
  else if(state.drawing&&tool==="freehand"){state.points.push(p);render();}
  else if(state.drawing&&tool==="eraser"){eraseLine(state.lastPointer,p);state.lastPointer=p;render();}
});
canvas.addEventListener("pointerup",()=>{const tool=$("tool").value;if(state.moving){state.moving=false;pushHistory();queueAutoSave();}else if(state.drawing&&tool==="freehand")completePolygon();else if(state.drawing&&tool==="eraser"){state.drawing=false;state.instancesByKey.set(annotationKey(),instances().filter(x=>x.mask.some(Boolean)));pushHistory();render();queueAutoSave();}});
canvas.addEventListener("dblclick",()=>{if($("tool").value==="polygon")completePolygon();});
canvas.addEventListener("contextmenu",(event)=>{event.preventDefault();if($("tool").value==="polygon")completePolygon();});
canvas.addEventListener("wheel",(event)=>{if(!mediaItem())return;event.preventDefault();if(event.ctrlKey){const old=state.scale;state.scale=Math.max(.1,Math.min(10,state.scale*(event.deltaY<0?1.1:.9)));const rect=canvas.getBoundingClientRect(),mx=event.clientX-rect.left,my=event.clientY-rect.top;state.offsetX=mx-(mx-state.offsetX)*state.scale/old;state.offsetY=my-(my-state.offsetY)*state.scale/old;state.fit=false;}else if(event.shiftKey)state.offsetX-=event.deltaY;else state.offsetY-=event.deltaY;render();},{passive:false});

async function activateOutputDirectory(handle){state.outputDir=handle;await loadManifest();await loadSavedMasksForCurrent();render();status(`Output: ${handle.name}`);queueAutoSave();}
async function chooseOutput(){try{const previous=await recalledHandle("output"),options={id:"ultrai-output",mode:"readwrite"};if(previous)options.startIn=previous;const handle=await window.showDirectoryPicker(options);await rememberHandle("output",handle);await activateOutputDirectory(handle);}catch(error){if(error.name!=="AbortError")alert(error.message);}}
async function restoreOutputDirectory(){const handle=await recalledHandle("output");if(!handle)return;try{if(await handle.queryPermission({mode:"readwrite"})==="granted")await activateOutputDirectory(handle);}catch{}}
async function maskBlob(instance){const out=document.createElement("canvas");out.width=state.sourceWidth;out.height=state.sourceHeight;const oc=out.getContext("2d"),image=oc.createImageData(out.width,out.height);for(let i=0;i<instance.mask.length;i++){const v=instance.mask[i]?255:0,p=i*4;image.data[p]=v;image.data[p+1]=v;image.data[p+2]=v;image.data[p+3]=255;}oc.putImageData(image,0,0);return new Promise(resolve=>out.toBlob(resolve,"image/png"));}
async function writeFile(dir,name,data){const handle=await dir.getFileHandle(name,{create:true});const stream=await handle.createWritable();await stream.write(data);await stream.close();}
async function saveMaskSet(output,list,frameName=""){
  const expected=new Set();for(const instance of list){const name=`${frameName}${safeClass(instance.className)}_${String(instance.id).padStart(3,"0")}.png`;expected.add(name);await writeFile(output,name,await maskBlob(instance));}
  for await(const [name,handle] of output.entries())if(handle.kind==="file"&&name.startsWith(frameName)&&name.endsWith(".png")&&!expected.has(name))await output.removeEntry(name);return list.length;
}
async function saveCurrentMaskSet(){
  if(!state.outputDir||!mediaItem())return 0;const item=mediaItem(),folderName=item.name.replace(/\.[^.]+$/,"")||(item.type==="video"?"video":"image"),output=await state.outputDir.getDirectoryHandle(folderName,{create:true}),frameName=item.type==="video"?`frame_${String(state.frameIndex).padStart(6,"0")}_`:"";return saveMaskSet(output,instances(),frameName);
}
function queueAutoSave(){
  if(!state.outputDir||!mediaItem())return;startAutoSave();
}
function startAutoSave(){
  autoSavePromise=autoSavePromise.then(async()=>{try{const saved=await saveCurrentMaskSet();await saveManifest();status(`Autosaved ${saved} mask(s)`);}catch(error){status(`Autosave failed: ${error.message}`);}});
}
async function flushAutoSave(){
  await autoSavePromise;
}
async function saveMasks(){
  if(!state.outputDir)return alert("Choose an output folder first.");const item=mediaItem();if(!item)return;await flushAutoSave();
  const output=await state.outputDir.getDirectoryHandle(item.name.replace(/\.[^.]+$/,"")||(item.type==="video"?"video":"image"),{create:true});
  let saved=0;
  if(item.type==="video"){
    const keyStart=`${item.id}:frame:`;
    for(const [key,list] of state.instancesByKey){
      if(!key.startsWith(keyStart))continue;const frame=Number(key.slice(keyStart.length));saved+=await saveMaskSet(output,list,`frame_${String(frame).padStart(6,"0")}_`);
    }
  }else saved=await saveMaskSet(output,instances());
  await saveManifest();status(`Saved ${saved} mask(s)`);
}
async function saveManifest(){
  if(!state.outputDir)return;const byName=new Map(state.manifestVideos.map(item=>[item.video_name,item]));for(const item of state.media.filter(x=>x.type==="video")){byName.set(item.name,{...(byName.get(item.name)||{}),video_name:item.name,location_label:item.location||""});}const videos=[...byName.values()];
  const manifest={version:3,location_set:state.locations,label_set:state.nerves,anatomy_set:state.anatomy,videos};await writeFile(state.outputDir,"nerve_manifest.json",JSON.stringify(manifest,null,2));
}
async function loadManifest(){
  if(!state.outputDir)return;try{const h=await state.outputDir.getFileHandle("nerve_manifest.json"),data=JSON.parse(await(await h.getFile()).text());state.locations=[...new Set([...DEFAULT_LOCATIONS,...(data.location_set||[])])];state.nerves=[...new Set([...DEFAULT_NERVES,...(data.label_set||[])])];state.anatomy=[...new Set([...DEFAULT_ANATOMY,...(data.anatomy_set||[])])];state.manifestVideos=Array.isArray(data.videos)?data.videos:[];for(const item of state.media.filter(x=>x.type==="video")){const entry=state.manifestVideos.find(x=>x.video_name===item.name);item.location=entry?.location_label||null;}state.location=mediaItem()?.location||null;rebuildAllChips();}catch{}
}
async function loadManifestLocation(videoName){
  await loadManifest();const item=mediaItem();if(item&&item.name===videoName){state.location=item.location||null;updateChipSelection();}
}
async function loadSavedMasksForCurrent(){
  if(!state.outputDir||!mediaItem()||state.instancesByKey.has(annotationKey()))return false;const item=mediaItem();
  try{
    const directory=await state.outputDir.getDirectoryHandle(item.name.replace(/\.[^.]+$/,"")||(item.type==="video"?"video":"image"));
    const frameName=item.type==="video"?`frame_${String(state.frameIndex).padStart(6,"0")}_`:"";
    const pattern=new RegExp(`^${frameName}(.+)_(\\d+)\\.png$`,"i"),loaded=[],ids=new Map();
    for await(const [name,handle] of directory.entries()){
      if(handle.kind!=="file"||!name.startsWith(frameName))continue;const match=name.match(pattern);if(!match)continue;const bitmap=await createImageBitmap(await handle.getFile());const temp=document.createElement("canvas");temp.width=state.sourceWidth;temp.height=state.sourceHeight;const tc=temp.getContext("2d");tc.drawImage(bitmap,0,0,temp.width,temp.height);bitmap.close();const pixels=tc.getImageData(0,0,temp.width,temp.height).data,mask=new Uint8Array(temp.width*temp.height);for(let i=0;i<mask.length;i++)mask[i]=pixels[i*4]>=128?1:0;const className=match[1].replaceAll("_"," ").toLowerCase(),id=Number(match[2]);loaded.push({className,id,mask});ids.set(className,Math.max(ids.get(className)||1,id+1));
    }
    state.instancesByKey.set(annotationKey(),loaded);state.nextIdsByKey.set(annotationKey(),ids);resetHistory();return loaded.length>0;
  }catch{return false;}
}

$("loadImages").onclick=loadImages;$("loadVideos").onclick=loadVideos;$("chooseOutput").onclick=chooseOutput;$("saveMasks").onclick=saveMasks;
$("clearData").onclick=async()=>{await flushAutoSave();pauseVideo();clearMediaUrls();state.media=[];state.mediaIndex=-1;state.instancesByKey.clear();rebuildMediaList();render();status("Cleared");};
$("mediaList").onchange=()=>openMedia(Number($("mediaList").value));$("prevMedia").onclick=()=>openMedia(state.mediaIndex-1);$("nextMedia").onclick=()=>openMedia(state.mediaIndex+1);
$("firstFrame").onclick=()=>setFrame(0);$("prevFrame").onclick=()=>setFrame(state.frameIndex-1);$("nextFrame").onclick=()=>setFrame(state.frameIndex+1);$("lastFrame").onclick=()=>setFrame(state.frameCount-1);$("frameSlider").oninput=()=>setFrame(Number($("frameSlider").value));$("play").onclick=togglePlayback;
$("fit").onclick=()=>{fitView();render();};$("undo").onclick=undo;$("redo").onclick=redo;$("clearMasks").onclick=()=>{state.instancesByKey.set(annotationKey(),[]);pushHistory();render();queueAutoSave();};
for(const id of ["showMasks","fillMasks","opacity"])$(id).oninput=()=>{saveToolSettings();render();};$("radius").oninput=saveToolSettings;$("tool").onchange=saveToolSettings;$("addLocation").onclick=()=>addLabel("location");$("addNerve").onclick=()=>addLabel("nerve");$("addAnatomy").onclick=()=>addLabel("anatomy");
document.addEventListener("keydown",(event)=>{if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="z"){event.preventDefault();event.shiftKey?redo():undo();}else if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="y"){event.preventDefault();redo();}else if(event.key==="ArrowLeft")setFrame(state.frameIndex-1);else if(event.key==="ArrowRight")setFrame(state.frameIndex+1);else if(event.key==="Escape")cancelDrawing();});
async function runSelfTest(){
  try{
    const previousSettings=localStorage.getItem(TOOL_SETTINGS_KEY);$("tool").value="eraser";$("showMasks").checked=false;$("fillMasks").checked=true;$("opacity").value="73";$("radius").value="19";saveToolSettings();$("tool").value="select";$("showMasks").checked=true;$("fillMasks").checked=false;$("opacity").value="50";$("radius").value="4";restoreToolSettings();if($("tool").value!=="eraser"||$("showMasks").checked||!$("fillMasks").checked||$("opacity").value!=="73"||$("radius").value!=="19")throw new Error("tool settings");if(previousSettings===null){localStorage.removeItem(TOOL_SETTINGS_KEY);$("tool").value="freehand";$("showMasks").checked=true;$("fillMasks").checked=false;$("opacity").value="50";$("radius").value="4";}else{localStorage.setItem(TOOL_SETTINGS_KEY,previousSettings);restoreToolSettings();}
    state.media=[{id:"selftest",name:"selftest.png",type:"image",element:null}];state.mediaIndex=0;state.sourceWidth=32;state.sourceHeight=32;
    state.activeClass="median";state.points=[{x:2,y:2},{x:12,y:2},{x:12,y:12},{x:2,y:12}];completePolygon();
    state.activeClass="artery";state.points=[{x:16,y:16},{x:28,y:16},{x:28,y:28},{x:16,y:28}];completePolygon();
    if(instances().length!==2||instances()[0].id!==1||instances()[1].id!==1)throw new Error("instance creation");
    if(classColor("median")===classColor("artery"))throw new Error("class colors");undo();if(instances().length!==1)throw new Error("undo");redo();if(instances().length!==2)throw new Error("redo");
    const source=new Uint8Array(32*32),target=new Uint8Array(32*32),moving=new Uint8Array(32*32);for(let y=8;y<14;y++)for(let x=7;x<13;x++){source[y*32+x]=200;target[(y+2)*32+x+3]=200;moving[y*32+x]=1;}const shifted=propagateMask(moving,source,target);if(!shifted[(10+2)*32+10+3])throw new Error("frame propagation");
    if(typeof UTIF!=="object"||typeof UTIF.decode!=="function")throw new Error("TIFF support");
    const blob=await maskBlob(instances()[0]);if(blob.type!=="image/png"||blob.size===0)throw new Error("mask PNG");document.body.dataset.selftest="pass";
  }catch(error){document.body.dataset.selftest=`fail:${error.message}`;}
}
window.addEventListener("resize",resizeCanvas);document.body.dataset.fileApi=String(typeof window.showOpenFilePicker==="function"&&typeof window.showDirectoryPicker==="function");restoreToolSettings();rebuildAllChips();resizeCanvas();restoreOutputDirectory();if(new URLSearchParams(location.search).has("selftest"))runSelfTest();

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Génère une version robuste de l'application LGV : GPS temps réel + PK métrique.

Entrée :  index_v3.html
Sortie :  index_v3_gps_corrige.html
"""
from pathlib import Path
import re

INPUT = Path('index_v3.html')
OUTPUT = Path('index_v3_gps_corrige.html')
VERSION = 'v3.95-gps-mapmatch'

if not INPUT.exists():
    raise SystemExit('ERREUR : index_v3.html introuvable')
s = INPUT.read_text(encoding='utf-8')

# Le script est idempotent : une version déjà corrigée est conservée telle quelle.
if VERSION in s:
    OUTPUT.write_text(s, encoding='utf-8', newline='\n')
    print('OK', OUTPUT, VERSION, '(déjà corrigé)')
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Version + cache bust PMTiles
# ---------------------------------------------------------------------------
s = re.sub(r"var APP_VER\s*=\s*'[^']+';", f"var APP_VER = '{VERSION}';", s, count=1)
s = re.sub(r"var V\s*=\s*'[^']+';", f"var V='{VERSION}';", s, count=1)
s = re.sub(r"var _pkMetricBase = new URL\('data/pk_metric\.pmtiles\?v=[^']+'",
           "var _pkMetricBase = new URL('data/pk_metric.pmtiles?v=395'", s, count=1)

# ---------------------------------------------------------------------------
# Couche technique PK métrique : indépendante des collisions des labels.
# ---------------------------------------------------------------------------
if "id:'pkm1-hit'" in s:
    s = re.sub(
        r"map\.addLayer\(\{id:'pkm1-hit',[\s\S]*?\n\s*paint:\{[\s\S]*?\}\}\);",
        "map.addLayer({id:'pkm1-hit',type:'circle',source:'pk_metric','source-layer':'pk_metric',\n      minzoom:19,maxzoom:22,\n      paint:{'circle-radius':2.5,'circle-color':'#4285f4','circle-opacity':0.01,'circle-stroke-width':0,'circle-stroke-opacity':0}});",
        s, count=1
    )
else:
    anchor = "addMetricTier('pkm1-symbol',20,22,9);"
    if anchor not in s:
        raise SystemExit('ERREUR : emplacement pkm1-hit introuvable')
    s = s.replace(anchor, anchor + "\n    map.addLayer({id:'pkm1-hit',type:'circle',source:'pk_metric','source-layer':'pk_metric',minzoom:19,maxzoom:22,paint:{'circle-radius':2.5,'circle-color':'#4285f4','circle-opacity':0.01,'circle-stroke-width':0,'circle-stroke-opacity':0}});", 1)

# ---------------------------------------------------------------------------
# Bloc GPS complet.
# ---------------------------------------------------------------------------
GPS_START = "// ══════════════════════════════════════════════════════════\n// LOCALISATION TEMPS RÉEL"
PK_START = "/* ── PK métrique : cache spatial des tuiles PMTiles réellement chargées. ── */"
# Compatibilité avec la variante actuelle sans point final.
PK_START_ALT = "/* ── PK métrique : cache spatial des tuiles PMTiles réellement chargées ── */"
PK_START_NEW = "/* ── PK métrique : cache + interrogation de la couche technique ── */"
start = s.find(GPS_START)
end = s.find(PK_START)
if end < 0:
    end = s.find(PK_START_ALT)
if end < 0:
    end = s.find(PK_START_NEW)
if start < 0 or end < 0 or end <= start:
    raise SystemExit(f'ERREUR : section GPS introuvable start={start} end={end}')

GPS = r'''// ══════════════════════════════════════════════════════════
// LOCALISATION TEMPS RÉEL – GPS robuste + caméra/navigation
// ══════════════════════════════════════════════════════════
function _isOnLGV(lat,lng){return lat>=44.3&&lat<=47.7&&lng>=-1.7&&lng<=1.3;}
function _restoreMapPos(){
  try{
    var p=JSON.parse(localStorage.getItem('lgv_map_pos')||'null');
    if(p&&isFinite(p.lat)&&isFinite(p.lng)&&_isOnLGV(p.lat,p.lng))map.jumpTo({center:[p.lng,p.lat],zoom:Math.min(Number(p.zoom)||8,12)});
  }catch(e){}
  var t=null;
  map.on('moveend',function(){clearTimeout(t);t=setTimeout(function(){try{var c=map.getCenter();localStorage.setItem('lgv_map_pos',JSON.stringify({lat:c.lat,lng:c.lng,zoom:map.getZoom()}));}catch(e){}},400);});
}

var _gpsWatch=null,_gpsFollowing=true,_gpsAutoStarted=false;
var _gpsLastRaw=null,_gpsLastAccepted=null,_gpsDisplayed=null;
var _gpsAnim=null,_gpsAnimRaf=null,_gpsCameraMoving=false,_gpsLastCameraTs=0;
var _gpsLastError='',_gpsMatch=null,_gpsLastPKUpdate=0;
var _gpsFilter={ready:false,lat0:0,lng0:0,x:0,y:0,vx:0,vy:0,ts:0};
var GPS_BLUE_MAX_ACCURACY=120;
var GPS_PK_MAX_ACCURACY=45;
var GPS_STALE_MS=6000;

function _gpsResetFilter(){_gpsFilter={ready:false,lat0:0,lng0:0,x:0,y:0,vx:0,vy:0,ts:0};}
function _gpsClamp(v,a,b){return Math.max(a,Math.min(b,v));}
function _gpsToXY(lat,lng){
  var f=_gpsFilter;
  return {x:(lng-f.lng0)*111320*Math.cos(f.lat0*Math.PI/180),y:(lat-f.lat0)*110540};
}
function _gpsToGeo(x,y){
  var f=_gpsFilter,lat=f.lat0+y/110540,den=111320*Math.cos(f.lat0*Math.PI/180);
  return {lat:lat,lng:den?f.lng0+x/den:f.lng0};
}
function _gpsAngleDiff(a,b){return Math.abs(((a-b+540)%360)-180);}
function _gpsBearing(lat1,lng1,lat2,lng2){
  var p=Math.PI/180,dL=(lng2-lng1)*p;
  var y=Math.sin(dL)*Math.cos(lat2*p);
  var x=Math.cos(lat1*p)*Math.sin(lat2*p)-Math.sin(lat1*p)*Math.cos(lat2*p)*Math.cos(dL);
  return (Math.atan2(y,x)*180/Math.PI+360)%360;
}
function _gpsCircle(lng,lat,r){
  var a=[],R=6378137,lr=lat*Math.PI/180,cl=Math.max(.15,Math.cos(lr));
  for(var i=0;i<=64;i++){var q=i/64*Math.PI*2,dx=Math.cos(q)*r,dy=Math.sin(q)*r;a.push([lng+dx/(R*cl)*180/Math.PI,lat+dy/R*180/Math.PI]);}
  return{type:'Feature',properties:{},geometry:{type:'Polygon',coordinates:[a]}};
}
function _gpsHeadingWedge(lng,lat,heading,r){
  if(!isFinite(heading))return emptyFC();
  var out=[[lng,lat]],R=6378137,cl=Math.max(.15,Math.cos(lat*Math.PI/180));
  for(var i=0;i<=16;i++){var h=(heading-30+60*i/16)*Math.PI/180,dx=Math.sin(h)*r,dy=Math.cos(h)*r;out.push([lng+dx/(R*cl)*180/Math.PI,lat+dy/R*180/Math.PI]);}
  out.push([lng,lat]);
  return{type:'Feature',properties:{},geometry:{type:'Polygon',coordinates:[out]}};
}
function _ensureGpsLayers(){
  if(!map.getSource('_gpsCircle'))map.addSource('_gpsCircle',{type:'geojson',data:emptyFC()});
  if(!map.getLayer('_gpsCircleFill'))map.addLayer({id:'_gpsCircleFill',type:'fill',source:'_gpsCircle',paint:{'fill-color':'#4285f4','fill-opacity':.13}});
  if(!map.getLayer('_gpsCircleLine'))map.addLayer({id:'_gpsCircleLine',type:'line',source:'_gpsCircle',paint:{'line-color':'#4285f4','line-width':1.1,'line-opacity':.5}});
  if(!map.getSource('_gpsHeading'))map.addSource('_gpsHeading',{type:'geojson',data:emptyFC()});
  if(!map.getLayer('_gpsHeadingFill'))map.addLayer({id:'_gpsHeadingFill',type:'fill',source:'_gpsHeading',paint:{'fill-color':'#4285f4','fill-opacity':.11}});
  if(!map.getSource('_gpsPoint'))map.addSource('_gpsPoint',{type:'geojson',data:emptyFC()});
  if(!map.getLayer('_gpsHalo'))map.addLayer({id:'_gpsHalo',type:'circle',source:'_gpsPoint',paint:{'circle-radius':15,'circle-color':'#4285f4','circle-opacity':.18}});
  if(!map.getLayer('_gpsDot'))map.addLayer({id:'_gpsDot',type:'circle',source:'_gpsPoint',paint:{'circle-radius':7,'circle-color':'#4285f4','circle-stroke-width':3,'circle-stroke-color':'#fff'}});
}
function _gpsAccuracyClass(a){return a<=5?'gps-good':a<=15?'gps-mid':'gps-bad';}
function _gpsAge(p){return p?Math.max(0,Date.now()-(p.receivedAt||p.timestamp)):Infinity;}
function _updateGpsStatus(p){
  var el=document.getElementById('gps-status');if(!el)return;
  if(!p){el.style.display='none';return;}
  var acc=Math.round(p.accuracy),age=_gpsAge(p),speed=p.speed!=null?p.speed:0;
  var sp=speed<.5?'0 km/h':Math.round(speed*3.6)+' km/h';
  var fix=age>GPS_STALE_MS?'stale':(p.accuracy<=10?'bon':'en cours');
  var m=_gpsMatch?' · '+(_gpsMatch.label||'PK'): ' · PK —';
  el.innerHTML='<div class="gps-title">📍 GPS temps réel</div><div class="gps-line"><span class="'+_gpsAccuracyClass(p.accuracy)+'">±'+acc+' m</span> · '+sp+'</div><div class="gps-line">'+(_gpsFollowing?'🎯 Suivi actif':'🖐 Carte libre')+' · '+fix+m+'</div>';
  el.style.display='block';
}
function _setLocateButton(){
  var b=document.getElementById('locate-btn');if(!b)return;
  if(_gpsWatch===null){b.style.background='white';b.style.color='#111';b.title='Activer le suivi GPS temps réel';}
  else if(_gpsFollowing){b.style.background='#4285f4';b.style.color='white';b.title='GPS actif · appuyer pour recentrer';}
  else{b.style.background='#dbeafe';b.style.color='#1565c0';b.title='GPS actif · appuyer pour reprendre le suivi';}
}

function _gpsFilterPosition(raw){
  var f=_gpsFilter,ts=raw.timestamp;
  if(!f.ready){f.ready=true;f.lat0=raw.lat;f.lng0=raw.lng;f.x=0;f.y=0;f.vx=0;f.vy=0;f.ts=ts;return {lat:raw.lat,lng:raw.lng,accuracy:raw.accuracy,timestamp:ts,receivedAt:Date.now(),speed:raw.speed,heading:raw.heading,rawLat:raw.lat,rawLng:raw.lng,innovation:0};}
  var dtRaw=(ts-f.ts)/1000;
  if(!isFinite(dtRaw)||dtRaw<=0)return null;
  if(dtRaw>8){_gpsResetFilter();return _gpsFilterPosition(raw);}
  var dt=_gpsClamp(dtRaw,.2,4),xy=_gpsToXY(raw.lat,raw.lng),px=f.x+f.vx*dt,py=f.y+f.vy*dt,rx=xy.x-px,ry=xy.y-py,res=Math.hypot(rx,ry);
  var prevAcc=_gpsLastAccepted?_gpsLastAccepted.accuracy:raw.accuracy,predSpeed=Math.hypot(f.vx,f.vy),inputSpeed=raw.speed!=null?raw.speed:0;
  var maxJump=Math.max(45,raw.accuracy*3,prevAcc*2.4,Math.max(inputSpeed,predSpeed)*dt+45);
  if(res>maxJump&&raw.accuracy>=Math.max(12,prevAcc*.82))return null;
  var q=_gpsClamp(18/(raw.accuracy+18),.12,.92),alpha=_gpsClamp(.18+.74*q,.20,.94);
  if(Math.max(inputSpeed,predSpeed)>2)alpha=_gpsClamp(alpha+.08,.28,.97);
  if(res>Math.max(15,raw.accuracy*1.4))alpha=_gpsClamp(alpha+.06,.25,.97);
  var beta=_gpsClamp(alpha*.52,.10,.50);
  f.x=px+alpha*rx;f.y=py+alpha*ry;f.vx=f.vx+(beta/dt)*rx;f.vy=f.vy+(beta/dt)*ry;
  if(raw.speed!=null&&raw.heading!=null&&raw.speed>=1){var h=raw.heading*Math.PI/180,svx=Math.sin(h)*raw.speed,svy=Math.cos(h)*raw.speed,blend=raw.speed>8?.28:.18;f.vx=f.vx*(1-blend)+svx*blend;f.vy=f.vy*(1-blend)+svy*blend;}
  else if(Math.max(inputSpeed,predSpeed)<.8){f.vx*=.45;f.vy*=.45;}
  f.ts=ts;
  var geo=_gpsToGeo(f.x,f.y),dsp=Math.hypot(f.vx,f.vy),dhead=dsp>1&&_gpsLastAccepted?_gpsBearing(_gpsLastAccepted.lat,_gpsLastAccepted.lng,geo.lat,geo.lng):null;
  return {lat:geo.lat,lng:geo.lng,accuracy:raw.accuracy,timestamp:ts,receivedAt:Date.now(),speed:raw.speed!=null?raw.speed:dsp,heading:raw.heading!=null?raw.heading:dhead,rawLat:raw.lat,rawLng:raw.lng,innovation:res};
}
function _gpsAccept(pos){
  if(!pos||!pos.coords)return null;
  var c=pos.coords,lat=Number(c.latitude),lng=Number(c.longitude),acc=Number(c.accuracy),ts=Number(pos.timestamp);
  if(!isFinite(lat)||!isFinite(lng)||!isFinite(acc)||acc<=0||acc>500)return null;
  if(!isFinite(ts)||ts<=0)ts=Date.now();
  var raw={lat:lat,lng:lng,accuracy:acc,timestamp:ts,speed:isFinite(Number(c.speed))&&Number(c.speed)>=0?Number(c.speed):null,heading:isFinite(Number(c.heading))&&Number(c.heading)>=0?Number(c.heading):null};
  if(_gpsLastRaw&&ts<=_gpsLastRaw.timestamp)return null;
  if(acc>GPS_BLUE_MAX_ACCURACY)return null;
  _gpsLastRaw=raw;
  if(_gpsLastAccepted&&(ts-_gpsLastAccepted.timestamp)>8000)_gpsResetFilter();
  var p=_gpsFilterPosition(raw);if(!p)return null;
  _gpsLastAccepted=p;if(!_gpsDisplayed)_gpsDisplayed={lat:p.lat,lng:p.lng};
  window.GPS={last:{lat:p.lat,lng:p.lng,accuracy:p.accuracy,timestamp:p.timestamp,receivedAt:p.receivedAt,speed:p.speed,heading:p.heading,rawLat:p.rawLat,rawLng:p.rawLng},raw:raw,filtered:p,match:_gpsMatch};
  return p;
}
function _animateGpsTo(p){
  _ensureGpsLayers();
  var from=_gpsDisplayed||{lat:p.lat,lng:p.lng},dist=_haversine(from.lat,from.lng,p.lat,p.lng),speed=p.speed!=null?p.speed:0,dur=dist>100?160:(speed>0.5?Math.max(180,Math.min(850,(dist/speed)*900)):320);
  _gpsAnim={start:performance.now(),duration:dur,from:from,to:{lat:p.lat,lng:p.lng},p:p};
  if(_gpsAnimRaf==null)_gpsAnimRaf=requestAnimationFrame(_gpsAnimationFrame);
}
function _gpsAnimationFrame(now){
  if(!_gpsAnim){_gpsAnimRaf=null;return;}
  var a=_gpsAnim,t=_gpsClamp((now-a.start)/a.duration,0,1),e=1-Math.pow(1-t,3),lat=a.from.lat+(a.to.lat-a.from.lat)*e,lng=a.from.lng+(a.to.lng-a.from.lng)*e,p=a.p;
  _gpsDisplayed={lat:lat,lng:lng};
  var s=map.getSource('_gpsPoint');if(s)s.setData({type:'Feature',properties:{accuracy:p.accuracy,heading:p.heading},geometry:{type:'Point',coordinates:[lng,lat]}});
  var sc=map.getSource('_gpsCircle');if(sc)sc.setData(_gpsCircle(lng,lat,Math.max(3,p.accuracy)));
  var sh=map.getSource('_gpsHeading');if(sh)sh.setData((p.heading!=null&&p.speed>=1)?_gpsHeadingWedge(lng,lat,p.heading,Math.max(15,Math.min(50,p.speed*1.4+12))):emptyFC());
  if(t>=1){_gpsAnim=null;_gpsAnimRaf=null;return;}
  _gpsAnimRaf=requestAnimationFrame(_gpsAnimationFrame);
}
function _gpsMoveCamera(p,force){
  if(!_gpsFollowing)return;
  var now=Date.now();if(!force&&now-_gpsLastCameraTs<250)return;
  _gpsLastCameraTs=now;var targetZoom=p.accuracy<=5?19.5:p.accuracy<=10?19:p.accuracy<=25?18.5:18,cur=map.getCenter(),moved=_haversine(cur.lat,cur.lng,p.lat,p.lng);
  _gpsCameraMoving=true;
  if(moved>200)map.jumpTo({center:[p.lng,p.lat],zoom:Math.max(map.getZoom(),targetZoom)});
  else map.easeTo({center:[p.lng,p.lat],zoom:Math.max(map.getZoom(),targetZoom),duration:220,essential:true});
  setTimeout(function(){_gpsCameraMoving=false;},300);
}
function _drawGps(p,force){_ensureGpsLayers();_animateGpsTo(p);_updateGpsStatus(p);_gpsMoveCamera(p,force);}

function _gpsDrawAndMatch(p){
  _drawGps(p,false);
  if(!_gpsFollowing)return;
  var now=Date.now();if(now-_gpsLastPKUpdate<180)return;_gpsLastPKUpdate=now;
  var match=_bestPkMetricMatch(p);
  if(match){_gpsMatch=match;if(window.GPS)window.GPS.match=match;_updateGpsStatus(p);}
  var stamp=p.timestamp;
  map.once('idle',function(){
    if(!_gpsLastAccepted||_gpsLastAccepted.timestamp!==stamp||!_gpsFollowing)return;
    _refreshPkMetricCache(_gpsLastAccepted);
    var m2=_bestPkMetricMatch(_gpsLastAccepted);
    if(m2){_gpsMatch=m2;if(window.GPS)window.GPS.match=m2;_updateGpsStatus(_gpsLastAccepted);}
  });
}
function startLiveLocation(follow){
  if(!navigator.geolocation){showNetToast('📍 Géolocalisation indisponible');return;}
  if(typeof follow==='boolean')_gpsFollowing=follow;
  if(_gpsWatch!==null){_setLocateButton();return;}
  _gpsResetFilter();_gpsLastRaw=null;_gpsLastAccepted=null;_gpsDisplayed=null;_gpsMatch=null;_gpsLastPKUpdate=0;if(typeof _pkMatchState!=='undefined')_pkMatchState.last=null;
  showNetToast('📡 Recherche du signal GPS haute précision…');
  _gpsWatch=navigator.geolocation.watchPosition(function(pos){var p=_gpsAccept(pos);if(p)_gpsDrawAndMatch(p);},function(e){
    _gpsLastError=e&&e.code===1?'permission refusée':e&&e.code===2?'position indisponible':'délai GPS';
    if(e&&e.code===1)stopLiveLocation(false);showNetToast('📍 '+_gpsLastError);_updateGpsStatus(_gpsLastAccepted);
  },{enableHighAccuracy:true,maximumAge:0,timeout:20000});
  _setLocateButton();
}
function stopLiveLocation(showMessage){
  if(_gpsWatch!==null){navigator.geolocation.clearWatch(_gpsWatch);_gpsWatch=null;}
  if(_gpsAnimRaf!=null)cancelAnimationFrame(_gpsAnimRaf);_gpsAnimRaf=null;_gpsAnim=null;_setLocateButton();
  if(showMessage!==false)showNetToast('📍 Suivi GPS arrêté');
}
function locateMe(){
  if(_gpsWatch===null){_gpsFollowing=true;startLiveLocation(true);return;}
  _gpsFollowing=true;_setLocateButton();if(_gpsLastAccepted)_gpsMoveCamera(_gpsLastAccepted,true);_updateGpsStatus(_gpsLastAccepted);showNetToast('📍 Suivi temps réel actif');
}
window.locateMe=locateMe;window.startLiveLocation=startLiveLocation;window.stopLiveLocation=stopLiveLocation;

map.on('dragstart',function(){if(!_gpsCameraMoving&&_gpsWatch!==null){_gpsFollowing=false;_setLocateButton();_updateGpsStatus(_gpsLastAccepted);}});
map.on('zoomstart',function(){if(!_gpsCameraMoving&&_gpsWatch!==null){_gpsFollowing=false;_setLocateButton();_updateGpsStatus(_gpsLastAccepted);}});
map.on('rotatestart',function(){if(!_gpsCameraMoving&&_gpsWatch!==null){_gpsFollowing=false;_setLocateButton();_updateGpsStatus(_gpsLastAccepted);}});
map.once('load',function(){
  _setLocateButton();
  if(!navigator.geolocation||_gpsAutoStarted)return;_gpsAutoStarted=true;
  if(navigator.permissions&&navigator.permissions.query)navigator.permissions.query({name:'geolocation'}).then(function(r){if(r.state==='granted')startLiveLocation(true);}).catch(function(){});
});
document.addEventListener('visibilitychange',function(){
  if(document.visibilityState!=='visible'||_gpsWatch===null)return;
  _gpsResetFilter();
  try{navigator.geolocation.getCurrentPosition(function(pos){var p=_gpsAccept(pos);if(p)_gpsDrawAndMatch(p);},function(){},{enableHighAccuracy:true,maximumAge:0,timeout:8000});}catch(e){}
});
window.addEventListener('pageshow',function(){
  if(_gpsWatch===null)return;
  if(!_gpsLastAccepted||_gpsAge(_gpsLastAccepted)>7000){
    _gpsResetFilter();
    try{navigator.geolocation.getCurrentPosition(function(pos){var p=_gpsAccept(pos);if(p)_gpsDrawAndMatch(p);},function(){},{enableHighAccuracy:true,maximumAge:0,timeout:8000});}catch(e){}
  }
});

function _getFreshHighAccuracyPosition(done){
  if(_gpsWatch!==null){
    var waited=Date.now(),ended=false;
    function waitFresh(){
      if(ended)return;
      if(_gpsLastAccepted&&_gpsAge(_gpsLastAccepted)<=3500&&_gpsLastAccepted.accuracy<=GPS_PK_MAX_ACCURACY){ended=true;done(_gpsLastAccepted);return;}
      if(Date.now()-waited>=6500){ended=true;done(null);return;}
      setTimeout(waitFresh,180);
    }
    waitFresh();
    return;
  }
  if(!navigator.geolocation){done(null);return;}
  _gpsResetFilter();_gpsLastRaw=null;_gpsLastAccepted=null;_gpsDisplayed=null;_gpsMatch=null;
  if(typeof _pkMatchState!=='undefined')_pkMatchState.last=null;
  var samples=[],wid=null,finished=false,started=Date.now();
  function finish(){
    if(finished)return;finished=true;
    if(wid!==null){navigator.geolocation.clearWatch(wid);wid=null;}
    if(!samples.length){done(null);return;}
    samples.sort(function(a,b){return a.accuracy-b.accuracy;});
    var best=samples[0],moving=(best.speed||0)>1.5;
    if(!moving&&samples.length>=3){
      var n=Math.min(6,samples.length),la=0,lo=0,w=0,aa=0;
      for(var i=0;i<n;i++){var q=samples[i],ww=1/Math.max(1,q.accuracy*q.accuracy);la+=q.lat*ww;lo+=q.lng*ww;w+=ww;aa+=q.accuracy;}
      best={lat:la/w,lng:lo/w,accuracy:aa/n,timestamp:Date.now(),receivedAt:Date.now(),speed:0,heading:null,averaged:n};
    }
    done(best);
  }
  wid=navigator.geolocation.watchPosition(function(pos){var p=_gpsAccept(pos);if(!p)return;samples.push(p);if(p.accuracy<=7||Date.now()-started>=2800)finish();},function(){finish();},{enableHighAccuracy:true,maximumAge:0,timeout:7000});
  setTimeout(finish,7200);
}
function findNearest(){
  if(!navigator.geolocation){showNetToast('📍 Géolocalisation non disponible');return;}
  if(!map.getSource('pk_metric')||!map.getLayer('pkm1-hit')){openDetail('📍 PK le plus proche','<div style="padding:16px">PK métrique indisponible.</div>');return;}
  openDetail('📍 Éléments LGV proches','<div style="padding:18px;text-align:center;color:#888"><p style="font-size:14px;margin:0">📡 Position GPS haute précision…</p><p style="font-size:11px;margin-top:8px">Mesure fraîche obligatoire avant le calcul PK.</p></div>');
  _getFreshHighAccuracyPosition(function(p){
    if(!p){closeDetail();showNetToast('📍 Position GPS indisponible ou précision insuffisante');return;}
    _gpsFollowing=true;_setLocateButton();_drawGps(p,true);
    _gpsCameraMoving=true;map.jumpTo({center:[p.lng,p.lat],zoom:Math.max(map.getZoom(),20.5)});setTimeout(function(){_gpsCameraMoving=false;},120);
    var start=Date.now(),done=false;
    function run(){
      if(done)return;
      _refreshPkMetricCache(p);
      var m=_bestPkMetricMatch(p);
      if(m){done=true;_finishFindNearest(p.lat,p.lng,p.accuracy);return;}
      if(Date.now()-start>=6500){done=true;_finishFindNearest(p.lat,p.lng,p.accuracy);return;}
      setTimeout(run,250);
    }
    map.once('idle',run);setTimeout(run,350);
  });
}
window.findNearest=findNearest;
'''
s = s[:start] + GPS + "\n" + s[end:]

# ---------------------------------------------------------------------------
# Bloc PK métrique : cache + candidats autour du GPS + score de continuité.
# ---------------------------------------------------------------------------
ps=s.find(PK_START_ALT)
if ps<0: ps=s.find(PK_START)
pe=s.find("// ══════════════════════════════════════════════════════════\n// DETECTION PK / ACCES / PAM LE PLUS PROCHE",ps)
if ps<0 or pe<0 or pe<=ps:
    raise SystemExit(f'ERREUR : bloc PK métrique introuvable ps={ps} pe={pe}')

PK = r'''/* ── PK métrique : cache + interrogation de la couche technique ── */
var _pkMetricCache=new Map(),_pkMetricLastRefresh=0,_pkMetricLastCenter=null;
var _pkMatchState={last:null};
function _pkMetricKey(f){
  var co=f.geometry&&f.geometry.coordinates||[],p=f.properties||{},pk=p._pk_label!=null?p._pk_label:p.pk;
  return co.length>=2?co[0].toFixed(7)+'|'+co[1].toFixed(7)+'|'+String(pk||''):'';
}
function _refreshPkMetricCache(center){
  if(!map.getSource('pk_metric'))return 0;
  var now=Date.now();if(now-_pkMetricLastRefresh<180)return 0;_pkMetricLastRefresh=now;
  var fs=[];
  try{fs=map.querySourceFeatures('pk_metric',{sourceLayer:'pk_metric'});fs.forEach(function(f){if(!f.geometry||f.geometry.type!=='Point')return;var k=_pkMetricKey(f);if(!k)return;var old=_pkMetricCache.get(k);if(old)old.seenAt=now;else _pkMetricCache.set(k,{props:Object.assign({},f.properties),geom:{type:'Point',coordinates:f.geometry.coordinates.slice()},seenAt:now});});}catch(e){console.warn('PK métrique cache:',e);}
  _pkMetricLastCenter=center||_pkMetricLastCenter;
  if(center)_pkMetricCache.forEach(function(v,k){if(_haversine(center.lat,center.lng,v.geom.coordinates[1],v.geom.coordinates[0])>2200&&now-v.seenAt>4000)_pkMetricCache.delete(k);});
  if(_pkMetricCache.size>16000){var a=[];_pkMetricCache.forEach(function(v,k){a.push([k,v.seenAt]);});a.sort(function(x,y){return x[1]-y[1];});for(var i=0;i<Math.max(0,a.length-12000);i++)_pkMetricCache.delete(a[i][0]);}
  return fs.length;
}
function _parsePKMeters(lbl){
  if(lbl==null)return -1;var s=String(lbl).trim(),m=s.match(/^(\d+)\s*\+\s*(\d{1,3})$/);if(m)return parseInt(m[1],10)*1000+parseInt(m[2],10);var n=Number(s);return isFinite(n)?Math.round(n):-1;
}
function _pkMetricLabel(p){
  var raw=p&&p._pk_label!=null?p._pk_label:(p&&p.pk!=null?p.pk:null);if(raw==null||raw==='')return '';
  var t=String(raw);if(t.indexOf('+')>=0)return t;var n=Number(raw);if(!isFinite(n))return t;return Math.floor(n/1000)+'+'+('000'+Math.round(n%1000)).slice(-3);
}
function _metricRenderedCandidates(lat,lng){
  if(!map.getLayer('pkm1-hit'))return[];
  try{
    var pt=map.project([lng,lat]),acc=(_gpsLastRaw&&isFinite(_gpsLastRaw.accuracy))?_gpsLastRaw.accuracy:20,radiusM=Math.max(25,Math.min(100,acc*2.4+10)),mpp=Math.max(.05,156543.03392*Math.cos(lat*Math.PI/180)/Math.pow(2,map.getZoom())),rp=Math.max(20,Math.min(500,radiusM/mpp));
    return map.queryRenderedFeatures([[pt.x-rp,pt.y-rp],[pt.x+rp,pt.y+rp]],{layers:['pkm1-hit']});
  }catch(e){return[];}
}
function _candidatePool(lat,lng){
  var a=_metricRenderedCandidates(lat,lng),b=[],seen={},out=[];
  _pkMetricCache.forEach(function(f){if(_haversine(lat,lng,f.geom.coordinates[1],f.geom.coordinates[0])<=110)b.push(f);});
  a.concat(b).forEach(function(f){var k=_pkMetricKey(f);if(k&&!seen[k]){seen[k]=1;out.push({props:f.properties||{},geom:f.geometry});}});
  return out;
}
function _pkCandidateScore(f,p){
  var co=f.geom.coordinates,d=_haversine(p.lat,p.lng,co[1],co[0]);
  if(p.accuracy>GPS_PK_MAX_ACCURACY)return null;
  var maxD=Math.max(15,Math.min(100,p.accuracy*2.45+8));if(d>maxD)return null;
  var voie=String(f.props._voie||f.props.voie||f.props.VOIE||'').trim().toUpperCase(),pk=_parsePKMeters(f.props._pk_label!=null?f.props._pk_label:f.props.pk),score=d,last=_pkMatchState.last;
  if(last){
    var dt=_gpsClamp((p.timestamp-last.timestamp)/1000,.25,5),speed=p.speed!=null?p.speed:last.speed||0,expected=Math.max(0,Math.min(140,speed*dt));
    if(last.voie&&voie){
      if(last.voie!==voie)score+=25;
      else if(last.pkMeters>=0&&pk>=0){
        var delta=pk-last.pkMeters,absDelta=Math.abs(delta),allowed=Math.max(80,expected*3+60);
        if(absDelta>allowed)score+=(absDelta-allowed)*2.5;
        if(expected>6&&p.heading!=null&&absDelta>4){var bearing=_gpsBearing(last.lat,last.lng,co[1],co[0]),ad=_gpsAngleDiff(bearing,p.heading);if(ad>45)score+=(ad-45)*.8;}
      }
    }
    var geoD=_haversine(last.lat,last.lng,co[1],co[0]),geoAllowed=Math.max(90,expected*3+80);if(geoD>geoAllowed)score+=(geoD-geoAllowed)*1.6;
  }
  return {f:f,d:d,score:score,voie:voie,pkMeters:pk};
}
function _bestPkMetricMatch(p){
  if(!p||p.accuracy>GPS_PK_MAX_ACCURACY)return null;
  _refreshPkMetricCache(p);var pool=_candidatePool(p.lat,p.lng),cand=[];
  pool.forEach(function(f){var x=_pkCandidateScore(f,p);if(x)cand.push(x);});
  cand.sort(function(a,b){return a.score-b.score;});if(!cand.length)return null;
  var best=cand[0],last=_pkMatchState.last;
  if(last&&last.voie&&best.voie&&best.voie!==last.voie){var same=cand.find(function(x){return x.voie===last.voie;});if(same&&best.score+10>=same.score)best=same;}
  var co=best.f.geom.coordinates,label=_pkMetricLabel(best.f.props);
  var out={lat:co[1],lng:co[0],d:best.d,score:best.score,voie:best.voie,label:label,pkMeters:best.pkMeters,props:best.f.props,geom:best.f.geom,timestamp:p.timestamp,speed:p.speed};
  _pkMatchState.last=out;return out;
}
function _nearestPkMetric(lat,lng,maxDistance){
  _refreshPkMetricCache({lat:lat,lng:lng});
  var p={lat:lat,lng:lng,accuracy:Math.min(GPS_PK_MAX_ACCURACY,Math.max(5,maxDistance/2.5)),timestamp:Date.now(),speed:0,heading:null};
  var m=_bestPkMetricMatch(p);return m?{f:{props:m.props,geom:m.geom},d:m.d}:null;
}
'''
s=s[:ps]+PK+'\n'+s[pe:]

# ---------------------------------------------------------------------------
# _finishFindNearest : PK métrique en premier, sans interpolation trompeuse.
# ---------------------------------------------------------------------------
fs=s.find('function _finishFindNearest(lat,lng,acc){')
fe=s.find('  /* Détecter branche + côté depuis la voie du PK le plus proche */',fs)
if fs<0 or fe<0:raise SystemExit('ERREUR : _finishFindNearest introuvable')
finish = r'''function _finishFindNearest(lat,lng,acc){
  var live={lat:lat,lng:lng,accuracy:acc,timestamp:Date.now(),speed:(_gpsLastAccepted&&_gpsLastAccepted.speed)||0,heading:(_gpsLastAccepted&&_gpsLastAccepted.heading!=null)?_gpsLastAccepted.heading:null};
  var metric=_bestPkMetricMatch(live),npk=null,pkPrec='';
  if(metric){npk={f:{props:metric.props,geom:metric.geom},d:metric.d};pkPrec='PK métrique 1 m · map-matching';}
  if(!npk&&_pkHData&&_pkHData.features&&_pkHData.features.length){
    var hs=_pkHData.features.map(function(f){return{f:{props:f.properties,geom:f.geometry},d:_haversine(lat,lng,f.geometry.coordinates[1],f.geometry.coordinates[0])};});
    hs.sort(function(a,b){return a.d-b.d;});if(hs.length){npk=hs[0];pkPrec='secours PK 100 m';}
  }
  if(!npk&&layerData.pk.length){npk=_nearestOf(lat,lng,layerData.pk);if(npk)pkPrec='secours PK 1 km';}
'''
s=s[:fs]+finish+s[fe:]

# ---------------------------------------------------------------------------
# Libellés de précision : la précision GPS n'est jamais annoncée comme 1 m.
# ---------------------------------------------------------------------------
s=s.replace("(pkPrec.indexOf('1 m')>=0?'#2e7d32':'#c62828')+';font-weight:700",
            "(pkPrec.indexOf('PK métrique')===0?'#2e7d32':'#c62828')+';font-weight:700",1)
s=s.replace("if(pkPrec&&pkPrec.indexOf('1 m')<0)html+='<div style=\"font-size:10.5px;color:#c62828;font-weight:700;margin-top:3px\">⚠ PK métrique indisponible : valeur de secours, pas une localisation au mètre.</div>';",
            "if(pkPrec.indexOf('PK métrique')!==0)html+='<div style=\"font-size:10.5px;color:#c62828;font-weight:700;margin-top:3px\">⚠ PK métrique indisponible : valeur de secours. Précision GPS : ±'+Math.round(acc)+' m.</div>';",1)

# ---------------------------------------------------------------------------
# Supprime toute ancienne implémentation de findNearest encore présente après
# la section DETECTION : il ne doit exister qu'un seul handler.
# ---------------------------------------------------------------------------
det=s.find("// ══════════════════════════════════════════════════════════\n// DETECTION PK / ACCES / PAM LE PLUS PROCHE")
if det>=0:
    old=s.find('function findNearest(){',det)
    if old>=0:
        stop=s.find('/* Extrait le côté (1 ou 2)',old)
        if stop<0: raise SystemExit('ERREUR : ancienne fonction findNearest non délimitée')
        s=s[:old]+s[stop:]

# ---------------------------------------------------------------------------
# Vérifications minimales mais strictes.
# ---------------------------------------------------------------------------
for marker in [
    VERSION,"id:'pkm1-hit'",'GPS_PK_MAX_ACCURACY','_gpsResetFilter();',
    'function _gpsDrawAndMatch(p){','function findNearest(){','function _bestPkMetricMatch(p){',
    '_metricRenderedCandidates(lat,lng)','visibilitychange','PK métrique 1 m · map-matching'
]:
    if marker not in s:raise SystemExit('ERREUR verification: '+marker)

OUTPUT.write_text(s,encoding='utf-8',newline='\n')
print('OK',OUTPUT,VERSION)

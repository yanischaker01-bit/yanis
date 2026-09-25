#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corrige la localisation GPS de index_v3.html et produit index_v3_gps_corrige.html."""
from pathlib import Path
import re

SRC = Path('index_v3.html')
DST = Path('index_v3_gps_corrige.html')

if not SRC.exists():
    raise SystemExit('ERREUR: placez ce script dans le meme dossier que index_v3.html')

html = SRC.read_text(encoding='utf-8')

# Version applicative pour invalider proprement les anciennes ressources.
html = re.sub(r"var APP_VER\s*=\s*'[^']+';", "var APP_VER = 'v3.73-gps';", html, count=1)
html = re.sub(r"var V\s*=\s*'v[^']+';", "var V='v3.73-gps';", html, count=1)

# Bouton GPS avec identifiant stable.
html = re.sub(
    r'<button(?:\s+id="locate-btn")?\s+class="fab fab-white"\s+title="[^"]*"\s+onclick="locateMe\(\)">📍</button>',
    '<button id="locate-btn" class="fab fab-white" title="Activer ou libérer le suivi GPS en temps réel" onclick="locateMe()">📍</button>',
    html,
    count=1
)

# Style visuel du suivi GPS.
GPS_CSS = r'''
/* GPS temps reel */
#locate-btn.gps-active{background:#0078d4;color:white;box-shadow:0 0 0 4px rgba(0,120,212,.22),0 3px 10px rgba(0,0,0,.30)}
#locate-btn.gps-free{background:#dbeafe;color:#1565c0}
#locate-btn.gps-active::after{content:"";position:absolute;inset:-5px;border:2px solid rgba(0,120,212,.45);border-radius:50%;animation:gpsPulse 1.8s ease-out infinite;pointer-events:none}
@keyframes gpsPulse{0%{transform:scale(.8);opacity:1}100%{transform:scale(1.35);opacity:0}}
'''
if '#locate-btn.gps-active' not in html:
    html = html.replace('</style>', GPS_CSS + '\n</style>', 1)

GPS_BLOCK = r'''// ══════════════════════════════════════════════════════════
// LOCALISATION GPS TEMPS REEL - FIREFOX / CHROME ANDROID
// ══════════════════════════════════════════════════════════
function _isOnLGV(lat,lng){return lat>=44.3&&lat<=47.7&&lng>=-1.7&&lng<=1.3;}

var GPS={
  watchId:null,running:false,following:false,last:null,best:null,
  accepted:0,rejected:0,lastDraw:0,listeners:[],restartTimer:null
};

function _restoreMapPos(){
  /* L'ancien centre est seulement une vue de carte, jamais une position GPS. */
  try{
    var p=JSON.parse(localStorage.getItem('lgv_map_pos')||'null');
    if(p&&p.lat&&p.lng&&_isOnLGV(p.lat,p.lng))
      map.jumpTo({center:[p.lng,p.lat],zoom:Math.min(Number(p.zoom)||8,12)});
  }catch(e){}
  var t=null;
  map.on('moveend',function(){
    if(GPS.following)return;
    clearTimeout(t);
    t=setTimeout(function(){
      try{var c2=map.getCenter();localStorage.setItem('lgv_map_pos',JSON.stringify({lat:c2.lat,lng:c2.lng,zoom:map.getZoom()}));}catch(e){}
    },700);
  });
}

function _gpsAge(pos){return pos&&pos.timestamp?Math.max(0,Date.now()-Number(pos.timestamp)):Infinity;}
function _gpsRaw(pos){
  var c=pos.coords;
  return{lat:+c.latitude,lng:+c.longitude,accuracy:+c.accuracy,
    timestamp:+pos.timestamp||Date.now(),speed:isFinite(+c.speed)?Math.max(0,+c.speed):null,
    heading:isFinite(+c.heading)?+c.heading:null};
}
function _gpsValidate(pos){
  if(!pos||!pos.coords)return{ok:false,reason:'position absente'};
  var p=_gpsRaw(pos);
  if(!isFinite(p.lat)||!isFinite(p.lng))return{ok:false,reason:'coordonnees invalides'};
  if(!isFinite(p.accuracy)||p.accuracy<=0)return{ok:false,reason:'precision inconnue'};
  if(_gpsAge(pos)>10000)return{ok:false,reason:'position ancienne '+Math.round(_gpsAge(pos)/1000)+' s'};
  if(p.accuracy>150)return{ok:false,reason:'precision insuffisante ±'+Math.round(p.accuracy)+' m'};
  if(GPS.last){
    var dt=Math.max(.2,(p.timestamp-GPS.last.timestamp)/1000);
    var d=_haversine(GPS.last.lat,GPS.last.lng,p.lat,p.lng);
    var allowed=Math.max(80,(p.accuracy+GPS.last.accuracy)*2,90*dt+50);
    /* Ne rejette un saut que s'il n'apporte pas une meilleure precision. */
    if(d>allowed&&p.accuracy>=GPS.last.accuracy)return{ok:false,reason:'saut GPS incoherent'};
  }
  return{ok:true,p:p};
}
function _gpsCircle(lng,lat,r){
  var a=[],R=6378137,lr=lat*Math.PI/180;
  for(var i=0;i<=64;i++){var q=i/64*Math.PI*2,dx=Math.cos(q)*r,dy=Math.sin(q)*r;a.push([lng+dx/(R*Math.cos(lr))*180/Math.PI,lat+dy/R*180/Math.PI]);}
  return{type:'Feature',properties:{},geometry:{type:'Polygon',coordinates:[a]}};
}
function _ensureGpsLayers(){
  if(!map.getSource('_gpsAccuracy'))map.addSource('_gpsAccuracy',{type:'geojson',data:emptyFC()});
  if(!map.getLayer('_gpsAccuracyFill'))map.addLayer({id:'_gpsAccuracyFill',type:'fill',source:'_gpsAccuracy',paint:{'fill-color':'#4285f4','fill-opacity':.14}});
  if(!map.getLayer('_gpsAccuracyLine'))map.addLayer({id:'_gpsAccuracyLine',type:'line',source:'_gpsAccuracy',paint:{'line-color':'#4285f4','line-width':1.2,'line-opacity':.55}});
  if(!map.getSource('_gpsPoint'))map.addSource('_gpsPoint',{type:'geojson',data:emptyFC()});
  if(!map.getLayer('_gpsHalo'))map.addLayer({id:'_gpsHalo',type:'circle',source:'_gpsPoint',paint:{'circle-radius':14,'circle-color':'#4285f4','circle-opacity':.18}});
  if(!map.getLayer('_gpsDot'))map.addLayer({id:'_gpsDot',type:'circle',source:'_gpsPoint',paint:{'circle-radius':7,'circle-color':'#4285f4','circle-stroke-width':3,'circle-stroke-color':'#fff'}});
}
function _drawGps(p,force){
  _ensureGpsLayers();
  map.getSource('_gpsPoint').setData({type:'Feature',properties:{accuracy:p.accuracy},geometry:{type:'Point',coordinates:[p.lng,p.lat]}});
  map.getSource('_gpsAccuracy').setData(_gpsCircle(p.lng,p.lat,Math.max(3,p.accuracy)));
  var now=Date.now();
  if(GPS.following&&(force||now-GPS.lastDraw>700)){
    GPS.lastDraw=now;
    var z=p.accuracy<=10?19:p.accuracy<=25?18.5:p.accuracy<=50?18:17;
    map.easeTo({center:[p.lng,p.lat],zoom:Math.max(map.getZoom(),z),duration:450,essential:true});
  }
}
function _setLocateButton(){
  var b=document.getElementById('locate-btn');if(!b)return;
  b.classList.toggle('gps-active',GPS.running&&GPS.following);
  b.classList.toggle('gps-free',GPS.running&&!GPS.following);
  b.title=!GPS.running?'Activer le suivi GPS en temps reel':GPS.following?'Suivi actif, appuyer pour liberer la carte':'GPS actif, appuyer pour recentrer';
}
function _gpsSuccess(pos){
  var v=_gpsValidate(pos);
  if(!v.ok){GPS.rejected++;console.warn('GPS refuse:',v.reason);showNetToast('📡 GPS en attente : '+v.reason);return;}
  var p=v.p;GPS.last=p;GPS.accepted++;
  if(!GPS.best||p.accuracy<GPS.best.accuracy)GPS.best=p;
  _drawGps(p,false);
  var speed=p.speed!==null?' · '+Math.round(p.speed*3.6)+' km/h':'';
  showNetToast('📍 GPS temps reel ±'+Math.round(p.accuracy)+' m'+speed+(GPS.following?'':' · carte libre'));
  GPS.listeners.slice().forEach(function(fn){try{fn(p);}catch(e){console.error(e);}});
}
function _gpsError(e){
  var msg=e&&e.code===1?'autorisation refusee':e&&e.code===2?'position indisponible':e&&e.code===3?'delai GPS depasse':'erreur GPS';
  if(e&&e.code===1)stopRealtimeGPS(false);
  showNetToast('📍 '+msg);
}
function startRealtimeGPS(follow){
  if(!navigator.geolocation){showNetToast('📍 Geolocalisation non disponible');return false;}
  if(typeof follow==='boolean')GPS.following=follow;
  if(GPS.running){_setLocateButton();if(GPS.last&&GPS.following)_drawGps(GPS.last,true);return true;}
  GPS.running=true;GPS.best=null;GPS.last=null;GPS.accepted=0;GPS.rejected=0;GPS.lastDraw=0;
  showNetToast('🛰 Recherche GPS haute precision…');
  GPS.watchId=navigator.geolocation.watchPosition(_gpsSuccess,_gpsError,{enableHighAccuracy:true,timeout:30000,maximumAge:0});
  _setLocateButton();return true;
}
function stopRealtimeGPS(showMessage){
  if(GPS.watchId!==null)try{navigator.geolocation.clearWatch(GPS.watchId);}catch(e){}
  GPS.watchId=null;GPS.running=false;GPS.following=false;clearTimeout(GPS.restartTimer);_setLocateButton();
  if(showMessage!==false)showNetToast('⏹ Suivi GPS arrete');
}
function locateMe(){
  if(!GPS.running){GPS.following=true;startRealtimeGPS(true);return;}
  GPS.following=!GPS.following;
  if(GPS.following&&GPS.last)_drawGps(GPS.last,true);
  _setLocateButton();showNetToast(GPS.following?'📍 Suivi temps reel actif':'📍 GPS actif, carte libre');
}
window.locateMe=locateMe;window.startRealtimeGPS=startRealtimeGPS;window.stopRealtimeGPS=stopRealtimeGPS;

/* Firefox Android peut suspendre le watcher en arriere-plan. On le recree au retour. */
document.addEventListener('visibilitychange',function(){
  if(document.visibilityState!=='visible'||!GPS.running)return;
  var follow=GPS.following;
  if(GPS.watchId!==null)try{navigator.geolocation.clearWatch(GPS.watchId);}catch(e){}
  GPS.watchId=null;GPS.running=false;GPS.last=null;GPS.best=null;
  clearTimeout(GPS.restartTimer);GPS.restartTimer=setTimeout(function(){startRealtimeGPS(follow);},250);
});
map.once('load',function(){_setLocateButton();});
'''

start_marker = '// ══════════════════════════════════════════════════════════\n// POSITION MAP + GÉOLOCALISATION'
alt_marker = '// ══════════════════════════════════════════════════════════\n// LOCALISATION TEMPS REEL'
end_marker = '// ══════════════════════════════════════════════════════════\n// DETECTION PK / ACCES / PAM LE PLUS PROCHE'

start = html.find(start_marker)
if start < 0:
    start = html.find(alt_marker)
end = html.find(end_marker, start if start >= 0 else 0)
if start < 0 or end < 0:
    raise SystemExit('ERREUR: section GPS introuvable dans index_v3.html')
html = html[:start] + GPS_BLOCK + '\n' + html[end:]

# Remplacement complet de findNearest, sans seconde localisation concurrente.
new_find = r'''function findNearest(){
  if(!navigator.geolocation){showNetToast('📍 Geolocalisation non disponible');return;}
  openDetail('📍 Éléments LGV proches','<div style="padding:20px;text-align:center;color:#888"><p style="font-size:14px;margin:0">🛰 Acquisition GPS precise…</p><p style="font-size:10px;margin-top:8px;color:#aaa">Les positions anciennes et imprecises sont ignorees.</p></div>');
  var done=false,timer=null;
  function cleanup(){GPS.listeners=GPS.listeners.filter(function(fn){return fn!==onFix;});if(timer){clearTimeout(timer);timer=null;}}
  function commit(p,warn){
    if(done)return;done=true;cleanup();GPS.following=true;_drawGps(p,true);_setLocateButton();
    if(warn)showNetToast('⚠ Calcul avec une precision de ±'+Math.round(p.accuracy)+' m');
    var fired=false;
    function run(){if(fired)return;fired=true;map.off('idle',run);_finishFindNearest(p.lat,p.lng,p.accuracy);}
    map.on('idle',run);setTimeout(run,3500);
  }
  function onFix(p){if(p.accuracy<=35)commit(p,false);}
  if(GPS.last&&Date.now()-GPS.last.timestamp<=5000&&GPS.last.accuracy<=35){commit(GPS.last,false);return;}
  GPS.listeners.push(onFix);startRealtimeGPS(false);
  timer=setTimeout(function(){
    if(done)return;
    var p=GPS.best;
    if(p&&Date.now()-p.timestamp<=10000&&p.accuracy<=100)commit(p,true);
    else{done=true;cleanup();closeDetail();showNetToast('📍 Position trop imprecise. Placez-vous a ciel ouvert.');}
  },30000);
}
window.findNearest=findNearest;'''

pattern = re.compile(r'function findNearest\(\)\{.*?\n\}\nwindow\.findNearest=findNearest;', re.S)
html, n = pattern.subn(new_find, html, count=1)
if n != 1:
    raise SystemExit('ERREUR: fonction findNearest introuvable')

DST.write_text(html, encoding='utf-8', newline='\n')
print(f'OK: {DST.resolve()}')

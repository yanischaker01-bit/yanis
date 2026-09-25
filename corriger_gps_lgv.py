#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correction GPS LGV SEA v3.75-gps-fresh.

Entree : index_v3.html
Sortie : index_v3_gps_corrige.html

Le script remplace uniquement la logique GPS et la fonction findNearest().
"""
from pathlib import Path
import re

SRC = Path("index_v3.html")
DST = Path("index_v3_gps_corrige.html")
VERSION = "v3.75-gps-fresh"

if not SRC.exists():
    raise SystemExit(
        "ERREUR : index_v3.html est introuvable. "
        "Placez corriger_gps_lgv.py dans le meme dossier."
    )

html = SRC.read_text(encoding="utf-8")

# Version applicative
html = re.sub(
    r"var\s+APP_VER\s*=\s*'[^']+';",
    f"var APP_VER = '{VERSION}';",
    html,
    count=1,
)
html = re.sub(
    r"var\s+V\s*=\s*'[^']+';",
    f"var V='{VERSION}';",
    html,
    count=1,
)

# Bouton de localisation
button_pattern = re.compile(
    r'<button(?=[^>]*onclick="locateMe\(\)")(?=[^>]*class="fab fab-white")[^>]*>📍</button>'
)
button = (
    '<button id="locate-btn" class="fab fab-white" '
    'title="Activer ou liberer le suivi GPS en temps reel" '
    'onclick="locateMe()">📍</button>'
)
html, count = button_pattern.subn(button, html, count=1)
if count != 1:
    raise SystemExit("ERREUR : bouton locateMe() introuvable.")

# Style du bouton GPS
css = r'''
/* GPS temps reel */
#locate-btn.gps-active{background:#0078d4;color:white;box-shadow:0 0 0 4px rgba(0,120,212,.22),0 3px 10px rgba(0,0,0,.30)}
#locate-btn.gps-free{background:#dbeafe;color:#1565c0}
#locate-btn.gps-active::after{content:"";position:absolute;inset:-5px;border:2px solid rgba(0,120,212,.45);border-radius:50%;animation:gpsPulse 1.8s ease-out infinite;pointer-events:none}
@keyframes gpsPulse{0%{transform:scale(.8);opacity:1}100%{transform:scale(1.35);opacity:0}}
'''
if "#locate-btn.gps-active" not in html:
    html = html.replace("</style>", css + "\n</style>", 1)

# Nouvelle logique GPS
GPS_JS = r'''// ══════════════════════════════════════════════════════════
// LOCALISATION GPS TEMPS REEL STABILISEE
// ══════════════════════════════════════════════════════════
function _isOnLGV(lat,lng){return lat>=44.3&&lat<=47.7&&lng>=-1.7&&lng<=1.3;}

var GPS={
  watchId:null,refreshTimer:null,watchdogTimer:null,restartTimer:null,
  running:false,following:false,requestPending:false,
  last:null,best:null,accepted:0,rejected:0,lastDraw:0,lastReceivedAt:0,
  sequence:0,lastAcceptedSequence:0,listeners:[]
};

function _restoreMapPos(){
  /* Ne restaure plus l'ancien centre, qui donnait l'impression d'une position GPS en cache. */
  try{localStorage.removeItem('lgv_map_pos');}catch(e){}
}

function _gpsAge(pos){
  return pos&&pos.timestamp?Math.max(0,Date.now()-Number(pos.timestamp)):Infinity;
}

function _gpsRaw(pos,sequence){
  var c=pos.coords;
  return{
    lat:Number(c.latitude),lng:Number(c.longitude),accuracy:Number(c.accuracy),
    timestamp:Number(pos.timestamp)||Date.now(),receivedAt:Date.now(),
    speed:c.speed!==null&&isFinite(Number(c.speed))?Math.max(0,Number(c.speed)):null,
    heading:c.heading!==null&&isFinite(Number(c.heading))?Number(c.heading):null,
    sequence:Number(sequence)||0
  };
}

function _gpsValidate(pos,sequence){
  if(!pos||!pos.coords)return{ok:false,reason:'position absente'};
  var p=_gpsRaw(pos,sequence);
  if(!isFinite(p.lat)||!isFinite(p.lng))return{ok:false,reason:'coordonnees invalides'};
  if(!isFinite(p.accuracy)||p.accuracy<=0)return{ok:false,reason:'precision inconnue'};
  var age=_gpsAge(pos);
  if(age>5000)return{ok:false,reason:'position ancienne '+Math.round(age/1000)+' s'};
  if(p.accuracy>80)return{ok:false,reason:'precision insuffisante ±'+Math.round(p.accuracy)+' m'};
  if(p.sequence>0&&p.sequence<GPS.lastAcceptedSequence)return{ok:false,reason:'reponse GPS depassee'};

  if(GPS.last){
    var dt=Math.max(.2,(p.timestamp-GPS.last.timestamp)/1000);
    var distance=_haversine(GPS.last.lat,GPS.last.lng,p.lat,p.lng);
    var maximum=Math.max(40,100*dt+p.accuracy+GPS.last.accuracy+25);
    if(distance>maximum){
      var correction=p.accuracy<=20&&p.accuracy<GPS.last.accuracy*.45;
      if(!correction)return{ok:false,reason:'saut incoherent de '+Math.round(distance)+' m'};
    }
    if(distance<=Math.max(15,GPS.last.accuracy)&&p.accuracy>GPS.last.accuracy*1.8)
      return{ok:false,reason:'mesure moins precise'};
    if(distance<.5&&Math.abs(p.accuracy-GPS.last.accuracy)<.5)
      return{ok:false,reason:'mesure identique'};
  }
  return{ok:true,p:p};
}

function _gpsCircle(lng,lat,r){
  var coordinates=[],R=6378137,latRad=lat*Math.PI/180;
  for(var i=0;i<=64;i++){
    var a=i/64*Math.PI*2,dx=Math.cos(a)*r,dy=Math.sin(a)*r;
    coordinates.push([lng+dx/(R*Math.cos(latRad))*180/Math.PI,lat+dy/R*180/Math.PI]);
  }
  return{type:'Feature',properties:{},geometry:{type:'Polygon',coordinates:[coordinates]}};
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
  map.getSource('_gpsPoint').setData({type:'Feature',properties:{accuracy:p.accuracy,heading:p.heading},geometry:{type:'Point',coordinates:[p.lng,p.lat]}});
  map.getSource('_gpsAccuracy').setData(_gpsCircle(p.lng,p.lat,Math.max(3,p.accuracy)));
  var now=Date.now();
  if(GPS.following&&(force||now-GPS.lastDraw>800)){
    GPS.lastDraw=now;
    var z=p.accuracy<=10?19:p.accuracy<=20?18.5:p.accuracy<=40?18:17.5;
    map.easeTo({center:[p.lng,p.lat],zoom:Math.max(map.getZoom(),z),duration:450,essential:true});
  }
}

function _setLocateButton(){
  var b=document.getElementById('locate-btn');if(!b)return;
  b.classList.toggle('gps-active',GPS.running&&GPS.following);
  b.classList.toggle('gps-free',GPS.running&&!GPS.following);
  b.title=!GPS.running?'Activer le suivi GPS':GPS.following?'Suivi actif, appuyer pour liberer la carte':'GPS actif, appuyer pour recentrer';
}

function _gpsSuccess(pos,sequence){
  GPS.requestPending=false;GPS.lastReceivedAt=Date.now();
  var validation=_gpsValidate(pos,sequence),p;
  if(!validation.ok){GPS.rejected++;console.warn('GPS refuse :',validation.reason);return;}
  p=validation.p;GPS.last=p;GPS.accepted++;
  if(p.sequence>0)GPS.lastAcceptedSequence=Math.max(GPS.lastAcceptedSequence,p.sequence);
  if(!GPS.best||p.accuracy<GPS.best.accuracy)GPS.best=p;
  _drawGps(p,false);
  var speed=p.speed!==null?' · '+Math.round(p.speed*3.6)+' km/h':'';
  showNetToast('📍 GPS ±'+Math.round(p.accuracy)+' m'+speed+(GPS.following?'':' · carte libre'));
  GPS.listeners.slice().forEach(function(fn){try{fn(p);}catch(e){console.error(e);}});
}

function _gpsError(error){
  GPS.requestPending=false;
  var msg=error&&error.code===1?'autorisation refusee':error&&error.code===2?'position indisponible':error&&error.code===3?'delai GPS depasse':'erreur GPS';
  if(error&&error.code===1)stopRealtimeGPS(false);
  console.warn('GPS :',error);showNetToast('📍 '+msg);
}

function _gpsRequestFreshPosition(){
  if(!GPS.running||GPS.requestPending||document.visibilityState!=='visible')return;
  GPS.requestPending=true;var sequence=++GPS.sequence;
  navigator.geolocation.getCurrentPosition(
    function(pos){_gpsSuccess(pos,sequence);},
    function(error){GPS.requestPending=false;if(error&&error.code===1)_gpsError(error);},
    {enableHighAccuracy:true,timeout:8000,maximumAge:0}
  );
}

function _gpsStartWatch(){
  if(GPS.watchId!==null)try{navigator.geolocation.clearWatch(GPS.watchId);}catch(e){}
  GPS.watchId=navigator.geolocation.watchPosition(
    function(pos){_gpsSuccess(pos,0);},_gpsError,
    {enableHighAccuracy:true,timeout:15000,maximumAge:0}
  );
}

function startRealtimeGPS(follow){
  if(!navigator.geolocation){showNetToast('📍 Geolocalisation indisponible');return false;}
  if(typeof follow==='boolean')GPS.following=follow;
  if(GPS.running){_setLocateButton();if(GPS.last&&GPS.following)_drawGps(GPS.last,true);return true;}
  GPS.running=true;GPS.requestPending=false;GPS.last=null;GPS.best=null;GPS.accepted=0;GPS.rejected=0;GPS.lastDraw=0;GPS.lastReceivedAt=0;GPS.sequence=0;GPS.lastAcceptedSequence=0;
  showNetToast('🛰 Acquisition GPS reelle…');
  _gpsStartWatch();_gpsRequestFreshPosition();
  clearInterval(GPS.refreshTimer);
  GPS.refreshTimer=setInterval(_gpsRequestFreshPosition,5000);
  clearInterval(GPS.watchdogTimer);
  GPS.watchdogTimer=setInterval(function(){
    if(!GPS.running||document.visibilityState!=='visible')return;
    if(!GPS.lastReceivedAt||Date.now()-GPS.lastReceivedAt>15000){
      GPS.last=null;GPS.best=null;GPS.requestPending=false;_gpsStartWatch();_gpsRequestFreshPosition();showNetToast('🛰 Renouvellement du signal GPS…');
    }
  },5000);
  _setLocateButton();return true;
}

function stopRealtimeGPS(showMessage){
  if(GPS.watchId!==null)try{navigator.geolocation.clearWatch(GPS.watchId);}catch(e){}
  clearInterval(GPS.refreshTimer);clearInterval(GPS.watchdogTimer);clearTimeout(GPS.restartTimer);
  GPS.watchId=null;GPS.refreshTimer=null;GPS.watchdogTimer=null;GPS.restartTimer=null;
  GPS.running=false;GPS.following=false;GPS.requestPending=false;GPS.last=null;GPS.best=null;
  _setLocateButton();if(showMessage!==false)showNetToast('⏹ Suivi GPS arrete');
}

function locateMe(){
  if(!GPS.running){GPS.following=true;startRealtimeGPS(true);return;}
  GPS.following=!GPS.following;if(GPS.following&&GPS.last)_drawGps(GPS.last,true);
  _setLocateButton();showNetToast(GPS.following?'📍 Suivi temps reel actif':'📍 GPS actif, carte libre');
}
window.locateMe=locateMe;window.startRealtimeGPS=startRealtimeGPS;window.stopRealtimeGPS=stopRealtimeGPS;

document.addEventListener('visibilitychange',function(){
  if(document.visibilityState!=='visible'||!GPS.running)return;
  var follow=GPS.following;
  if(GPS.watchId!==null)try{navigator.geolocation.clearWatch(GPS.watchId);}catch(e){}
  clearInterval(GPS.refreshTimer);clearInterval(GPS.watchdogTimer);
  GPS.watchId=null;GPS.refreshTimer=null;GPS.watchdogTimer=null;GPS.running=false;GPS.requestPending=false;
  GPS.last=null;GPS.best=null;GPS.lastReceivedAt=0;
  clearTimeout(GPS.restartTimer);GPS.restartTimer=setTimeout(function(){startRealtimeGPS(follow);},300);
});
map.once('load',function(){try{localStorage.removeItem('lgv_map_pos');}catch(e){}_setLocateButton();});
'''

start_markers = [
    "// ══════════════════════════════════════════════════════════\n// POSITION MAP + GÉOLOCALISATION",
    "// ══════════════════════════════════════════════════════════\n// LOCALISATION TEMPS REEL",
    "// ══════════════════════════════════════════════════════════\n// LOCALISATION TEMPS RÉEL",
    "// ══════════════════════════════════════════════════════════\n// LOCALISATION GPS TEMPS REEL",
    "// ══════════════════════════════════════════════════════════\n// LOCALISATION GPS TEMPS RÉEL",
]
positions = [html.find(m) for m in start_markers if html.find(m) >= 0]
start = min(positions) if positions else -1
end_marker = "// ══════════════════════════════════════════════════════════\n// DETECTION PK / ACCES / PAM LE PLUS PROCHE"
end = html.find(end_marker, start if start >= 0 else 0)
if start < 0 or end < 0:
    raise SystemExit("ERREUR : section GPS introuvable.")
html = html[:start] + GPS_JS + "\n" + html[end:]

# findNearest reutilise strictement la position GPS validee
FIND_JS = r'''function findNearest(){
  if(!navigator.geolocation){showNetToast('📍 Geolocalisation indisponible');return;}
  openDetail('📍 Éléments LGV proches','<div style="padding:20px;text-align:center;color:#888"><p style="font-size:14px;margin:0">🛰 Acquisition GPS precise…</p><p style="font-size:10px;margin-top:8px;color:#aaa">Les positions anciennes et imprecises sont ignorees.</p></div>');
  var done=false,timer=null;
  function cleanup(){GPS.listeners=GPS.listeners.filter(function(fn){return fn!==onFix;});if(timer!==null){clearTimeout(timer);timer=null;}}
  function commit(p,warn){
    if(done)return;done=true;cleanup();GPS.following=true;_drawGps(p,true);_setLocateButton();
    if(warn)showNetToast('⚠ Calcul avec une precision de ±'+Math.round(p.accuracy)+' m');
    var ran=false;function calculate(){if(ran)return;ran=true;map.off('idle',calculate);_finishFindNearest(p.lat,p.lng,p.accuracy);}
    map.on('idle',calculate);setTimeout(calculate,3500);
  }
  function onFix(p){if(p.accuracy<=35&&Date.now()-p.receivedAt<=5000)commit(p,false);}
  if(GPS.last&&Date.now()-GPS.last.receivedAt<=5000&&GPS.last.accuracy<=35){commit(GPS.last,false);return;}
  GPS.listeners.push(onFix);startRealtimeGPS(false);_gpsRequestFreshPosition();
  timer=setTimeout(function(){
    if(done)return;var p=GPS.best;
    if(p&&Date.now()-p.receivedAt<=10000&&p.accuracy<=80)commit(p,true);
    else{done=true;cleanup();closeDetail();showNetToast('📍 Position trop imprecise. Placez-vous a ciel ouvert.');}
  },30000);
}
window.findNearest=findNearest;'''

html, count = re.subn(
    r"function\s+findNearest\(\)\s*\{.*?\n\}\nwindow\.findNearest\s*=\s*findNearest;",
    FIND_JS,
    html,
    count=1,
    flags=re.S,
)
if count != 1:
    raise SystemExit("ERREUR : fonction findNearest() introuvable.")

# Verification finale
required = [VERSION, "GPS.requestPending", "_gpsRequestFreshPosition", "maximumAge:0", "localStorage.removeItem('lgv_map_pos')"]
missing = [x for x in required if x not in html]
if missing:
    raise SystemExit("ERREUR verification finale : " + ", ".join(missing))

DST.write_text(html, encoding="utf-8", newline="\n")
print("OK :", DST)
print("Version :", VERSION)

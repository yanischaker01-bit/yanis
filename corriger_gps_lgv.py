#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Génère une version de l'application LGV avec suivi GPS filtré et map-matching PK.

Entrée :  index_v3.html
Sortie :  index_v3_gps_corrige.html

Principes :
- watchPosition() haute précision uniquement pour le suivi terrain ;
- filtre alpha-beta léger, adaptatif à l'exactitude GPS ;
- rejet des sauts manifestes sans figer le déplacement ;
- point bleu animé indépendamment de la caméra ;
- mode caméra actif/passif comme une appli de navigation ;
- cache PK métrique issu des tuiles PMTiles réellement chargées ;
- map-matching avec distance, continuité de voie, continuité de déplacement et cap ;
- pas de fausse précision : une PK 1 m reste une précision de données, pas une précision GPS ;
- fallback explicite vers PK 100 m puis 1 km uniquement si les PK métriques sont indisponibles.
"""
from pathlib import Path
import re
import shutil

INPUT = Path("index_v3.html")
OUTPUT = Path("index_v3_gps_corrige.html")
VERSION = "v3.90-gps-mapmatch"

if not INPUT.exists():
    raise SystemExit("ERREUR : index_v3.html introuvable")

s = INPUT.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Version applicative / cache
# ---------------------------------------------------------------------------
s = re.sub(r"var APP_VER\s*=\s*'[^']+';", f"var APP_VER = '{VERSION}';", s, count=1)
s = re.sub(r"var V\s*=\s*'[^']+';", f"var V='{VERSION}';", s, count=1)
s = re.sub(r"var _pkMetricBase = new URL\('data/pk_metric\.pmtiles\?v=[^']+'",
           "var _pkMetricBase = new URL('data/pk_metric.pmtiles?v=390'", s, count=1)

# ---------------------------------------------------------------------------
# Style + statut GPS
# ---------------------------------------------------------------------------
CSS_ANCHOR = """#net-toast{position:fixed;top:60px;left:50%;transform:translateX(-50%);z-index:10000;"""
CSS_INSERT = r'''
/* ── Statut GPS temps réel ── */
#gps-status{position:absolute;left:10px;bottom:max(36px,calc(env(safe-area-inset-bottom,0px)+10px));
  z-index:801;background:rgba(255,255,255,.96);color:#1a1a2e;border-radius:10px;
  box-shadow:0 2px 12px rgba(0,0,0,.25);padding:7px 10px;min-width:158px;
  font-size:10.5px;line-height:1.35;display:none;pointer-events:none}
#gps-status .gps-title{font-weight:800;font-size:10px;text-transform:uppercase;letter-spacing:.4px}
#gps-status .gps-line{margin-top:2px;white-space:nowrap}
#gps-status .gps-good{color:#2e7d32;font-weight:700}
#gps-status .gps-mid{color:#ef6c00;font-weight:700}
#gps-status .gps-bad{color:#c62828;font-weight:700}
'''
if "#gps-status{position:absolute" not in s:
    s = s.replace(CSS_ANCHOR, CSS_INSERT + "\n" + CSS_ANCHOR, 1)

HTML_ANCHOR = '<div id="net-toast"></div>'
if '<div id="gps-status">' not in s:
    s = s.replace(HTML_ANCHOR, HTML_ANCHOR + '\n<div id="gps-status"></div>', 1)

# ---------------------------------------------------------------------------
# GPS section : remplacement complet
# ---------------------------------------------------------------------------
GPS_START = "// ══════════════════════════════════════════════════════════\n// LOCALISATION TEMPS REEL"
GPS_END = "// ══════════════════════════════════════════════════════════\n// DETECTION PK / ACCES / PAM LE PLUS PROCHE"
start = s.find(GPS_START)
end = s.find(GPS_END)
if start < 0 or end < 0 or end <= start:
    raise SystemExit("ERREUR : section GPS introuvable")

GPS_BLOCK = r'''// ══════════════════════════════════════════════════════════
// LOCALISATION TEMPS RÉEL – GPS filtré + navigation type Field Maps
// ══════════════════════════════════════════════════════════
function _isOnLGV(lat,lng){return lat>=44.3&&lat<=47.7&&lng>=-1.7&&lng<=1.3;}
function _restoreMapPos(){
  try{var p=JSON.parse(localStorage.getItem('lgv_map_pos')||'null');if(p&&p.lat&&_isOnLGV(p.lat,p.lng))map.jumpTo({center:[p.lng,p.lat],zoom:Math.min(p.zoom,12)});}catch(e){}
  var t=null;map.on('moveend',function(){clearTimeout(t);t=setTimeout(function(){try{var c=map.getCenter();localStorage.setItem('lgv_map_pos',JSON.stringify({lat:c.lat,lng:c.lng,zoom:map.getZoom()}));}catch(e){}},400);});
}

/* ── État GPS centralisé ── */
var _gpsWatch=null;
var _gpsFollowing=true;
var _gpsAutoStarted=false;
var _gpsLastRaw=null;
var _gpsLastAccepted=null;
var _gpsDisplayed=null;
var _gpsAnim=null;
var _gpsAnimRaf=null;
var _gpsCameraMoving=false;
var _gpsLastCameraTs=0;
var _gpsFilter={ready:false,lat0:0,lng0:0,x:0,y:0,vx:0,vy:0,ts:0};
var _gpsLastError='';
var _gpsMatch=null;

/* fenêtre locale en mètres autour du premier point du filtre */
function _gpsToXY(lat,lng){
  var f=_gpsFilter,lat0=f.lat0*Math.PI/180;
  return {x:(lng-f.lng0)*111320*Math.cos(lat0),y:(lat-f.lat0)*110540};
}
function _gpsToGeo(x,y){
  var f=_gpsFilter,lat=f.lat0+y/110540,lng=f.lng0+x/(111320*Math.cos(f.lat0*Math.PI/180));
  return {lat:lat,lng:lng};
}
function _gpsClamp(v,a,b){return Math.max(a,Math.min(b,v));}
function _gpsAngleDiff(a,b){
  var d=Math.abs(((a-b+540)%360)-180);return d>180?360-d:d;
}
function _gpsBearing(lat1,lng1,lat2,lng2){
  var p=Math.PI/180,dL=(lng2-lng1)*p;
  var y=Math.sin(dL)*Math.cos(lat2*p);
  var x=Math.cos(lat1*p)*Math.sin(lat2*p)-Math.sin(lat1*p)*Math.cos(lat2*p)*Math.cos(dL);
  return (Math.atan2(y,x)*180/Math.PI+360)%360;
}
function _gpsCircle(lng,lat,r){
  var a=[],R=6378137,lr=lat*Math.PI/180;
  for(var i=0;i<=64;i++){var q=i/64*Math.PI*2,dx=Math.cos(q)*r,dy=Math.sin(q)*r;a.push([lng+dx/(R*Math.cos(lr))*180/Math.PI,lat+dy/R*180/Math.PI]);}
  return{type:'Feature',properties:{},geometry:{type:'Polygon',coordinates:[a]}};
}
function _gpsHeadingWedge(lng,lat,heading,r){
  if(!isFinite(heading))return emptyFC();
  var out=[[lng,lat]],R=6378137,lr=lat*Math.PI/180,startH=heading-28,endH=heading+28;
  for(var i=0;i<=14;i++){
    var h=(startH+(endH-startH)*(i/14))*Math.PI/180;
    var dx=Math.sin(h)*r,dy=Math.cos(h)*r;
    out.push([lng+dx/(R*Math.cos(lr))*180/Math.PI,lat+dy/R*180/Math.PI]);
  }
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
function _gpsAccuracyClass(acc){return acc<=5?'gps-good':acc<=15?'gps-mid':'gps-bad';}
function _updateGpsStatus(p){
  var el=document.getElementById('gps-status');if(!el)return;
  if(!p){el.style.display='none';return;}
  var age=Math.max(0,Date.now()-p.timestamp),acc=Math.round(p.accuracy);
  var cls=_gpsAccuracyClass(p.accuracy);
  var speed=p.speed!=null?p.speed:0;
  var sp=speed<1?'0 km/h':Math.round(speed*3.6)+' km/h';
  var fix=age>5000?'stale':(acc<=10?'bon':'en cours');
  var match=_gpsMatch?' · PK '+(_gpsMatch.label||'✓'):'';
  el.innerHTML='<div class="gps-title">📍 GPS temps réel</div>'+
    '<div class="gps-line"><span class="'+cls+'">±'+acc+' m</span> · '+sp+'</div>'+
    '<div class="gps-line">'+(_gpsFollowing?'🎯 Suivi actif':'🖐 Carte libre')+' · '+fix+match+'</div>';
  el.style.display='block';
}
function _setLocateButton(){
  var b=document.getElementById('locate-btn');if(!b)return;
  b.style.background=_gpsWatch===null?'white':(_gpsFollowing?'#4285f4':'#dbeafe');
  b.style.color=_gpsWatch===null?'#111':(_gpsFollowing?'white':'#1565c0');
  b.title=_gpsWatch===null?'Activer le suivi GPS temps réel':(_gpsFollowing?'Suivi GPS actif · appuyer pour libérer la carte':'Position active · appuyer pour recentrer');
}

/* ── Filtre alpha-beta adaptatif ──
 * Plus l'exactitude GPS est bonne, plus la mesure est suivie rapidement.
 * Les gros sauts restent filtrés, mais un véhicule en mouvement n'est jamais figé.
 */
function _gpsFilterPosition(raw){
  var f=_gpsFilter,xy=_gpsToXY(raw.lat,raw.lng),ts=raw.timestamp;
  if(!f.ready){
    f.ready=true;f.lat0=raw.lat;f.lng0=raw.lng;f.x=0;f.y=0;f.vx=0;f.vy=0;f.ts=ts;
    return {lat:raw.lat,lng:raw.lng,accuracy:raw.accuracy,timestamp:ts,speed:raw.speed,heading:raw.heading,rawLat:raw.lat,rawLng:raw.lng};
  }
  var dt=_gpsClamp((ts-f.ts)/1000,.2,5);
  var px=f.x+f.vx*dt,py=f.y+f.vy*dt;
  var rx=xy.x-px,ry=xy.y-py,res=Math.hypot(rx,ry);
  var prevAcc=_gpsLastAccepted?_gpsLastAccepted.accuracy:raw.accuracy;
  var predictedSpeed=Math.hypot(f.vx,f.vy);
  var maxJump=Math.max(50,raw.accuracy*3,prevAcc*2.5,(Math.max(raw.speed||0,predictedSpeed)*dt)+40);
  if(res>maxJump&&raw.accuracy>=prevAcc*0.85){
    return null;
  }
  var q=_gpsClamp(15/(raw.accuracy+15),.10,.95);
  var alpha=_gpsClamp(.22+.68*q, .18, .88);
  if((raw.speed||predictedSpeed)>2)alpha=_gpsClamp(alpha+.08,.25,.93);
  if(res>Math.max(20,raw.accuracy*1.5))alpha=_gpsClamp(alpha+.06,.25,.93);
  var beta=_gpsClamp(alpha*.55,.10,.48);
  f.x=px+alpha*rx;f.y=py+alpha*ry;
  f.vx=f.vx+(beta/dt)*rx;f.vy=f.vy+(beta/dt)*ry;

  /* Une vitesse/cap fiables servent à stabiliser la direction estimée. */
  if(raw.speed!=null&&raw.heading!=null&&raw.speed>=0){
    var h=raw.heading*Math.PI/180,mvx=Math.sin(h)*raw.speed,mvy=Math.cos(h)*raw.speed;
    var blend=(raw.speed>2)?.22:.10;
    f.vx=f.vx*(1-blend)+mvx*blend;f.vy=f.vy*(1-blend)+mvy*blend;
  }else if((raw.speed||0)<.8){f.vx*=.55;f.vy*=.55;}
  f.ts=ts;
  var geo=_gpsToGeo(f.x,f.y);
  return {lat:geo.lat,lng:geo.lng,accuracy:raw.accuracy,timestamp:ts,speed:raw.speed!=null?raw.speed:Math.hypot(f.vx,f.vy),heading:raw.heading,rawLat:raw.lat,rawLng:raw.lng,innovation:res};
}
function _gpsAccept(pos){
  var c=pos.coords,lat=+c.latitude,lng=+c.longitude,acc=+c.accuracy,ts=+pos.timestamp||Date.now();
  if(!isFinite(lat)||!isFinite(lng)||!isFinite(acc)||acc<=0||acc>500||Date.now()-ts>15000)return null;
  var raw={lat:lat,lng:lng,accuracy:acc,timestamp:ts,
    speed:isFinite(+c.speed)&&+c.speed>=0?+c.speed:null,
    heading:isFinite(+c.heading)&&+c.heading>=0?+c.heading:null};
  _gpsLastRaw=raw;
  var p=_gpsFilterPosition(raw);
  if(!p)return null;
  _gpsLastAccepted=p;
  if(!_gpsDisplayed)_gpsDisplayed={lat:p.lat,lng:p.lng};
  window.GPS={last:{lat:p.lat,lng:p.lng,accuracy:p.accuracy,timestamp:p.timestamp,receivedAt:Date.now(),speed:p.speed,heading:p.heading,rawLat:p.rawLat,rawLng:p.rawLng},raw:raw,filtered:p,match:_gpsMatch};
  return p;
}

/* Animation du point bleu : la position visuelle suit la cible sans saut brutal. */
function _animateGpsTo(p){
  _ensureGpsLayers();
  var from=_gpsDisplayed||{lat:p.lat,lng:p.lng},to={lat:p.lat,lng:p.lng};
  var dist=_haversine(from.lat,from.lng,to.lat,to.lng);
  var dur=_gpsAnim?Math.max(120,Math.min(360,(_gpsAnim.duration||220)*.85)):220;
  if(dist>80)dur=120;
  _gpsAnim={start:performance.now(),duration:dur,from:from,to:to,p:p};
  if(_gpsAnimRaf==null)_gpsAnimRaf=requestAnimationFrame(_gpsAnimationFrame);
}
function _gpsAnimationFrame(now){
  if(!_gpsAnim){_gpsAnimRaf=null;return;}
  var a=_gpsAnim,t=_gpsClamp((now-a.start)/a.duration,0,1),e=1-Math.pow(1-t,3);
  var lat=a.from.lat+(a.to.lat-a.from.lat)*e,lng=a.from.lng+(a.to.lng-a.from.lng)*e,p=a.p;
  _gpsDisplayed={lat:lat,lng:lng};
  var s=map.getSource('_gpsPoint');
  if(s)s.setData({type:'Feature',properties:{accuracy:p.accuracy,heading:p.heading},geometry:{type:'Point',coordinates:[lng,lat]}});
  var sc=map.getSource('_gpsCircle');
  if(sc)sc.setData(_gpsCircle(lng,lat,Math.max(3,p.accuracy)));
  var sh=map.getSource('_gpsHeading');
  if(sh)sh.setData((p.heading!=null&&((p.speed||0)>=1))?_gpsHeadingWedge(lng,lat,p.heading,Math.max(15,Math.min(45,(p.speed||0)*1.5+10))):emptyFC());
  if(t>=1){_gpsAnim=null;_gpsAnimRaf=null;return;}
  _gpsAnimRaf=requestAnimationFrame(_gpsAnimationFrame);
}

function _drawGps(p,force){
  _ensureGpsLayers();
  _animateGpsTo(p);
  _updateGpsStatus(p);
  var now=Date.now();
  if(_gpsFollowing&&(force||now-_gpsLastCameraTs>300)){
    _gpsLastCameraTs=now;
    var targetZoom=p.accuracy<=5?19.5:p.accuracy<=10?19:p.accuracy<=25?18.5:18;
    var cur=map.getCenter(),moved=_haversine(cur.lat,cur.lng,p.lat,p.lng);
    _gpsCameraMoving=true;
    if(moved>150){
      map.jumpTo({center:[p.lng,p.lat],zoom:Math.max(map.getZoom(),targetZoom)});
    }else{
      map.easeTo({center:[p.lng,p.lat],zoom:Math.max(map.getZoom(),targetZoom),duration:220,essential:true});
    }
    setTimeout(function(){_gpsCameraMoving=false;},280);
  }
}

function startLiveLocation(follow){
  if(!navigator.geolocation){showNetToast('📍 Géolocalisation indisponible');return;}
  if(typeof follow==='boolean')_gpsFollowing=follow;
  if(_gpsWatch!==null){_setLocateButton();return;}
  _gpsFilter={ready:false,lat0:0,lng0:0,x:0,y:0,vx:0,vy:0,ts:0};
  _gpsLastAccepted=null;_gpsLastRaw=null;_gpsMatch=null;
  showNetToast('📡 Recherche du signal GPS haute précision…');
  _gpsWatch=navigator.geolocation.watchPosition(function(pos){
    var p=_gpsAccept(pos);if(!p)return;
    _gpsDrawAndMatch(p);
  },function(e){
    _gpsLastError=e&&e.code===1?'permission refusée':e&&e.code===3?'délai GPS dépassé':'signal GPS indisponible';
    if(e&&e.code===1)stopLiveLocation(false);
    else showNetToast('📍 '+_gpsLastError);
    _updateGpsStatus(_gpsLastAccepted);
  },{enableHighAccuracy:true,maximumAge:0,timeout:20000});
  _setLocateButton();
}

function _gpsDrawAndMatch(p){
  /* Le point bleu reste GPS filtré. Le PK est un résultat séparé du map-matching. */
  _drawGps(p,false);
  var match=_bestPkMetricMatch(p);
  if(match){
    _gpsMatch=match;
    window.GPS.match=match;
    _updateGpsStatus(p);
  }
  /* Après le déplacement de caméra, de nouvelles tuiles PMTiles peuvent être disponibles.
     Rejoue le match une fois le viewport stabilisé pour éviter un retard de PK. */
  var stamp=p.timestamp;
  setTimeout(function(){
    if(!_gpsLastAccepted||_gpsLastAccepted.timestamp!==stamp)return;
    _refreshPkMetricCache(_gpsLastAccepted);
    var m2=_bestPkMetricMatch(_gpsLastAccepted);
    if(m2){_gpsMatch=m2;window.GPS.match=m2;_updateGpsStatus(_gpsLastAccepted);}
  },280);
}

function stopLiveLocation(showMessage){
  if(_gpsWatch!==null){navigator.geolocation.clearWatch(_gpsWatch);_gpsWatch=null;}
  _setLocateButton();
  if(showMessage!==false)showNetToast('📍 Suivi GPS arrêté');
}
function locateMe(){
  if(_gpsWatch===null){_gpsFollowing=true;startLiveLocation(true);}
  else{
    _gpsFollowing=!_gpsFollowing;
    if(_gpsFollowing&&_gpsLastAccepted)_drawGps(_gpsLastAccepted,true);
    _setLocateButton();
    showNetToast(_gpsFollowing?'📍 Suivi temps réel actif':'📍 Position active, carte libre');
  }
}
window.locateMe=locateMe;window.startLiveLocation=startLiveLocation;window.stopLiveLocation=stopLiveLocation;

/* Comme les applications de navigation : un déplacement manuel passe le suivi en mode passif. */
map.on('dragstart',function(){if(!_gpsCameraMoving&&_gpsWatch!==null){_gpsFollowing=false;_setLocateButton();_updateGpsStatus(_gpsLastAccepted);}});
map.once('load',function(){
  _setLocateButton();
  if(!navigator.geolocation||_gpsAutoStarted)return;_gpsAutoStarted=true;
  var mobile=matchMedia('(pointer:coarse)').matches||/Android|iPhone|iPad|Mobile/i.test(navigator.userAgent);
  if(!mobile)return;
  if(navigator.permissions&&navigator.permissions.query)navigator.permissions.query({name:'geolocation'}).then(function(r){if(r.state==='granted')startLiveLocation(true);}).catch(function(){});
});

/* Rafraîchit les tuiles PK métriques chargées et limite la taille du cache. */
function _gpsRefreshMetricAfterMove(){
  if(typeof _refreshPkMetricCache==='function')_refreshPkMetricCache(_gpsLastAccepted);
}

// Reprise propre après sortie de veille / changement d'onglet.
document.addEventListener('visibilitychange',function(){
  if(document.visibilityState==='visible'&&_gpsWatch!==null){
    try{navigator.geolocation.getCurrentPosition(function(pos){var p=_gpsAccept(pos);if(p)_gpsDrawAndMatch(p);},function(){},{enableHighAccuracy:true,maximumAge:0,timeout:8000});}catch(e){}
  }
});
window.addEventListener('pageshow',function(){
  if(_gpsWatch!==null&&_gpsLastAccepted&&Date.now()-_gpsLastAccepted.timestamp>7000){
    try{navigator.geolocation.getCurrentPosition(function(pos){var p=_gpsAccept(pos);if(p)_gpsDrawAndMatch(p);},function(){},{enableHighAccuracy:true,maximumAge:0,timeout:8000});}catch(e){}
  }
});
'''

s = s[:start] + GPS_BLOCK + "\n" + s[end:]

# ---------------------------------------------------------------------------
# PK helpers : remplacer _nearest PK métrique et _finishFindNearest.
# Insert juste avant la section de détection PK.
# ---------------------------------------------------------------------------
PK_START = s.find(GPS_END)
if PK_START < 0:
    raise SystemExit("ERREUR : section détection PK absente après remplacement GPS")

PK_HELPERS = r'''
/* ── PK métrique : cache spatial des tuiles PMTiles réellement chargées ── */
var _pkMetricCache=new Map();
var _pkMetricLastCenter=null;
var _pkMetricLastRefresh=0;
var _pkMatchState={last:null};
function _pkMetricKey(f){
  var co=f.geometry&&f.geometry.coordinates||[];
  return co.length>=2?co[0].toFixed(7)+'|'+co[1].toFixed(7)+'|'+String(f.properties&&(f.properties._pk_label!=null?f.properties._pk_label:f.properties.pk)||''):'';
}
function _refreshPkMetricCache(center){
  if(!map.getSource('pk_metric'))return 0;
  var now=Date.now();
  if(now-_pkMetricLastRefresh<250)return 0;
  _pkMetricLastRefresh=now;
  var added=0;
  try{
    var fs=map.querySourceFeatures('pk_metric',{sourceLayer:'pk_metric'});
    fs.forEach(function(f){
      if(!f.geometry||f.geometry.type!=='Point')return;
      var k=_pkMetricKey(f);if(!k)return;
      if(!_pkMetricCache.has(k)){
        _pkMetricCache.set(k,{props:Object.assign({},f.properties),geom:{type:'Point',coordinates:f.geometry.coordinates.slice()},seenAt:now});
        added++;
      }else{_pkMetricCache.get(k).seenAt=now;}
    });
  }catch(e){console.warn('PK metric cache:',e);}
  _pkMetricLastCenter=center||_pkMetricLastCenter;
  /* Garde une fenêtre raisonnable autour du GPS actuel. */
  if(center){
    var cLat=center.lat,cLng=center.lng;
    _pkMetricCache.forEach(function(v,k){
      var d=_haversine(cLat,cLng,v.geom.coordinates[1],v.geom.coordinates[0]);
      if(d>1800&&now-v.seenAt>2500)_pkMetricCache.delete(k);
    });
  }
  /* Garde-fou mémoire ; on conserve les entrées les plus récemment vues. */
  if(_pkMetricCache.size>14000){
    var arr=[];_pkMetricCache.forEach(function(v,k){arr.push({k:k,t:v.seenAt});});
    arr.sort(function(a,b){return a.t-b.t;});
    arr.slice(0,Math.max(0,arr.length-11000)).forEach(function(x){_pkMetricCache.delete(x.k);});
  }
  return added;
}
function _parsePKMeters(lbl){
  if(lbl==null)return -1;
  var s=String(lbl).trim(),m=s.match(/^(\d+)\s*\+\s*(\d{1,3})$/);
  if(m)return parseInt(m[1],10)*1000+parseInt(m[2],10);
  var n=Number(s);return isFinite(n)?Math.round(n):-1;
}
function _pkMetricLabel(p){
  var raw=p&&p._pk_label!=null?p._pk_label:(p&&p.pk!=null?p.pk:null);
  if(raw==null||raw==='')return '';
  var text=String(raw);if(text.indexOf('+')>=0)return text;
  var n=Number(raw);if(!isFinite(n))return text;
  var km=Math.floor(n/1000),m=Math.round(n-km*1000);
  return km+'+'+('000'+m).slice(-3);
}
function _pkCandidateScore(f,p){
  var co=f.geom.coordinates,d=_haversine(p.lat,p.lng,co[1],co[0]);
  if(p.accuracy>35)return null;
  var maxD=Math.max(12,Math.min(55,p.accuracy*1.7));
  if(d>maxD)return null;
  var voie=String(f.props._voie||f.props.voie||'').trim();
  var score=d;
  if(_pkMatchState.last){
    var lm=_pkMatchState.last,dt=Math.max(.5,Math.min(5,(p.timestamp-lm.timestamp)/1000));
    var travel=_haversine(lm.lat,lm.lng,co[1],co[0]);
    var expected=Math.max(0,Math.min(100,(p.speed!=null?p.speed:lm.speed||0)*dt));
    if(travel>expected+70)score+=(travel-(expected+70))*2.2;
    if(lm.voie&&voie&&lm.voie!==voie)score+=12;
    if(p.heading!=null&&travel>5){var br=_gpsBearing(lm.lat,lm.lng,co[1],co[0]),ad=_gpsAngleDiff(br,p.heading);if(ad>35)score+=(ad-35)*.22;}
  }
  return {f:f,d:d,score:score,voie:voie};
}
function _bestPkMetricMatch(p){
  _refreshPkMetricCache(p);
  var groups={};
  _pkMetricCache.forEach(function(f){
    var cand=_pkCandidateScore(f,p);if(!cand)return;
    var key=cand.voie||'_generic';
    if(!groups[key]||cand.score<groups[key].score)groups[key]=cand;
  });
  var candidates=Object.keys(groups).map(function(k){return groups[k];}).sort(function(a,b){return a.score-b.score;});
  if(!candidates.length)return null;
  var best=candidates[0],old=_pkMatchState.last;
  if(old&&old.voie&&best.voie&&old.voie!==best.voie){
    var same=groups[old.voie];
    /* Hystérésis : ne change de voie que si le nouveau candidat est nettement meilleur. */
    if(same&&best.score+8>=same.score)best=same;
  }
  var props=best.f.props,co=best.f.geom.coordinates,label=_pkMetricLabel(props);
  var out={lat:co[1],lng:co[0],d:best.d,score:best.score,voie:best.voie,
    label:label,pkMeters:_parsePKMeters(label),props:props,geom:best.f.geom,timestamp:p.timestamp,speed:p.speed};
  _pkMatchState.last=out;
  return out;
}
function _nearestPkMetric(lat,lng,maxDistance){
  _refreshPkMetricCache({lat:lat,lng:lng});
  var p={lat:lat,lng:lng,accuracy:Math.max(1,maxDistance/1.9),timestamp:Date.now(),speed:0,heading:null};
  var m=_bestPkMetricMatch(p);return m?{f:{props:m.props,geom:m.geom},d:m.d}:null;
}
'''

# Avoid duplicate helper insertion if generator is re-run on an already corrected input.
if '_pkMetricCache=new Map()' not in s:
    s = s[:PK_START] + PK_HELPERS + "\n" + s[PK_START:]

# ---------------------------------------------------------------------------
# Remplacement de findNearest + _finishFindNearest complet jusqu'à branche.
# ---------------------------------------------------------------------------
DET_START = s.find("function findNearest(){")
DET_END = s.find("  /* Détecter branche + côté depuis la voie du PK le plus proche */", DET_START)
if DET_START < 0 or DET_END < 0:
    raise SystemExit("ERREUR : fonctions findNearest/_finishFindNearest introuvables")

DETECTION = r'''function _getFreshHighAccuracyPosition(done){
  if(_gpsLastAccepted&&Date.now()-_gpsLastAccepted.timestamp<3500&&_gpsLastAccepted.accuracy<=40){
    done(_gpsLastAccepted);return;
  }
  if(!navigator.geolocation){done(null);return;}
  /* Hors suivi permanent, on repart d'un état neutre pour ne pas hériter d'un ancien trajet. */
  if(_gpsWatch===null){_gpsFilter={ready:false,lat0:0,lng0:0,x:0,y:0,vx:0,vy:0,ts:0};_gpsLastAccepted=null;_gpsMatch=null;}
  var samples=[],wid=null,start=Date.now(),best=null;
  function finish(){
    if(wid!==null){navigator.geolocation.clearWatch(wid);wid=null;}
    if(samples.length){
      samples.sort(function(a,b){return a.accuracy-b.accuracy;});
      best=samples[0];
      /* En stationnaire, moyenne simple de 3–5 positions = mode « averaging ». */
      var moving=(best.speed||0)>1.5;
      if(!moving&&samples.length>=3){
        var n=Math.min(5,samples.length),la=0,lo=0,w=0,aa=0;
        for(var i=0;i<n;i++){var q=samples[i],ww=1/Math.max(1,q.accuracy*q.accuracy);la+=q.lat*ww;lo+=q.lng*ww;w+=ww;aa+=q.accuracy;}
        best={lat:la/w,lng:lo/w,accuracy:aa/n,timestamp:Date.now(),speed:0,heading:null,averaged:n};
      }
      done(best);return;
    }
    done(null);
  }
  wid=navigator.geolocation.watchPosition(function(pos){
    var p=_gpsAccept(pos);if(!p)return;
    samples.push(p);
    if(!best||p.accuracy<best.accuracy)best=p;
    if(Date.now()-start>=2500||p.accuracy<=8)finish();
  },function(){finish();},{enableHighAccuracy:true,maximumAge:0,timeout:6500});
  setTimeout(function(){if(wid!==null)finish();},7000);
}
function findNearest(){
  if(!navigator.geolocation){showNetToast('📍 Géolocalisation non disponible');return;}
  openDetail('📍 Éléments LGV proches','<div style="padding:20px;text-align:center;color:#888">'+
    '<p style="font-size:14px;margin:0">📡 Position GPS haute précision…</p>'+
    '<p style="font-size:10px;margin-top:8px;line-height:1.5;color:#aaa">Le PK ne sera calculé qu\'avec une position fraîche et cohérente.</p></div>');
  _getFreshHighAccuracyPosition(function(p){
    if(!p){closeDetail();showNetToast('📍 Position GPS indisponible');return;}
    if(!_gpsFollowing){_gpsFollowing=true;_setLocateButton();}
    _drawGps(p,true);
    map.jumpTo({center:[p.lng,p.lat],zoom:Math.max(map.getZoom(),20)});
    var tries=0,done=false;
    function finish(){
      if(done)return;tries++;
      _refreshPkMetricCache(p);
      var metric=_bestPkMetricMatch(p);
      if(metric||tries>=8){done=true;_finishFindNearest(p.lat,p.lng,p.accuracy);return;}
      setTimeout(finish,350);
    }
    map.once('idle',finish);setTimeout(finish,1200);
  });
}
window.findNearest=findNearest;

/* Extrait le côté (1 ou 2) depuis n'importe quel nom de voie/axe */
function _sideNum(v){var m=String(v||'').match(/[12]/);return m?m[0]:'';}

/* Normalise un nom de voie/axe en {b:branche_canonique, s:côté} */
function _parseVoie(v){
  var raw=String(v||'').trim().toUpperCase();
  if(!raw)return{b:'',s:''};
  if(raw==='LGV1'||raw==='V1'||raw==='LGV V1')return{b:'lgv',s:'1'};
  if(raw==='LGV2'||raw==='V2'||raw==='LGV V2')return{b:'lgv',s:'2'};
  if(raw==='LGV'||/^LGV[\s]/.test(raw))return{b:'lgv',s:''};
  var mR=raw.match(/^R([A-Z]+?)([12])$/)||raw.match(/^R([12])([A-Z]+)$/);
  if(mR){var letters,digit;if(/^[12]$/.test(mR[1])){digit=mR[1];letters=mR[2];}else{letters=mR[1];digit=mR[2];}
    var RMAP={J:'JU',V:'VN'};return{b:(RMAP[letters]||letters).toLowerCase(),s:digit};}
  var m=raw.match(/^([A-Z]+?)([12])$/)||raw.match(/^([12])([A-Z]+)$/);
  if(!m)return{b:raw.toLowerCase(),s:''};
  var letters3,digit3;if(/^[12]$/.test(m[1])){digit3=m[1];letters3=m[2];}else{letters3=m[1];digit3=m[2];}
  return{b:letters3.toLowerCase(),s:digit3};
}

function _finishFindNearest(lat,lng,acc){
  var npk=null,pkPrec='';
  var live={lat:lat,lng:lng,accuracy:acc,timestamp:Date.now(),speed:(_gpsLastAccepted&&_gpsLastAccepted.speed)||0,heading:(_gpsLastAccepted&&_gpsLastAccepted.heading)||null};
  _refreshPkMetricCache(live);
  var metric=_bestPkMetricMatch(live);
  if(metric){
    npk={f:{props:metric.props,geom:metric.geom},d:metric.d};
    pkPrec='PK métrique 1 m · matchage LGV';
  }

  /* Pas d'interpolation artificielle : un PK 100 m reste 100 m. */
  if(!npk&&_pkHData&&_pkHData.features&&_pkHData.features.length){
    var candidates=[];
    _pkHData.features.forEach(function(f){
      if(!f.geometry||f.geometry.type!=='Point')return;
      var d=_haversine(lat,lng,f.geometry.coordinates[1],f.geometry.coordinates[0]);
      candidates.push({f:{props:f.properties,geom:f.geometry},d:d});
    });
    candidates.sort(function(a,b){return a.d-b.d;});
    if(candidates.length){npk=candidates[0];pkPrec='secours PK 100 m';}
  }
  if(!npk&&layerData.pk.length){npk=_nearestOf(lat,lng,layerData.pk);if(npk)pkPrec='secours PK 1 km';}

  /* Détecter branche + côté depuis la voie du PK le plus proche */
'''
# Note: the replacement intentionally starts at the old findNearest and ends immediately before the existing branch logic.
s = s[:DET_START] + DETECTION + s[DET_END + len("  /* Détecter branche + côté depuis la voie du PK le plus proche */") :]

# ---------------------------------------------------------------------------
# Ajustements ciblés dans le rendu de _finishFindNearest.
# ---------------------------------------------------------------------------
# Remplace le texte de précision et le formatage de label dans le bloc restant.
s = s.replace("html+='<p style=\"font-size:11px;color:#888;margin:0\">📍 GPS : ±'+Math.round(acc)+' m · PK : précision '+pkPrec+voieInfo+'</p>';",
              "html+='<p style=\"font-size:11px;color:'+(pkPrec.indexOf('1 m')>=0?'#2e7d32':'#c62828')+';font-weight:700;margin:0\">📍 GPS : ±'+Math.round(acc)+' m (≈95% · navigateur) · '+(pkPrec||'PK indisponible')+voieInfo+'</p>';\n  if(npk&&npk.d!=null)html+='<p style=\"font-size:10.5px;color:#666;margin:-5px 0 0\">Distance entre GPS filtré et PK : <b>'+_dStr(npk.d)+'</b></p>';", 1)

# Fausse mention de précision héritée supprimée / normalisée.
s = s.replace("var rawLbl=p._pk_label!=null?p._pk_label:(p.pk!=null?p.pk:null);",
              "var rawLbl=(pkPrec.indexOf('1 m')>=0)?_pkMetricLabel(p):(p._pk_label!=null?p._pk_label:(p.pk!=null?p.pk:null));", 1)

# Message de secours : explicite mais non alarmiste.
needle = "html+='<div style=\"font-size:18px;font-weight:700;color:#cc0000\">'+lbl+'</div>';"
rep = needle + "\n    if(pkPrec&&pkPrec.indexOf('1 m')<0)html+='<div style=\"font-size:10.5px;color:#c62828;font-weight:700;margin-top:3px\">⚠ PK métrique indisponible : valeur de secours, pas une localisation au mètre.</div>';"
if needle in s:
    s = s.replace(needle, rep, 1)

# ---------------------------------------------------------------------------
# Helper layer invisible pour garder les tuiles PK métriques chargées à z17+.
# ---------------------------------------------------------------------------
MARKER = "addMetricTier('pkm1-symbol',20,22,9);"
HELPER = MARKER + r'''
    /* Couche technique invisible : le navigateur charge les tuiles PK autour du mobile
       dès z17+, sans afficher les milliers de points métriques. */
    if(!map.getLayer('pkm-query')){
      map.addLayer({id:'pkm-query',type:'circle',source:'pk_metric','source-layer':'pk_metric',minzoom:17,maxzoom:22,
        paint:{'circle-radius':1,'circle-opacity':0,'circle-stroke-opacity':0}});
    }'''
if "id:'pkm-query'" not in s:
    if MARKER not in s:
        raise SystemExit("ERREUR : couche PK métrique 1 m introuvable")
    s = s.replace(MARKER, HELPER, 1)

# ---------------------------------------------------------------------------
# Vérifications / sortie
# ---------------------------------------------------------------------------
required = [
    VERSION,
    "enableHighAccuracy:true,maximumAge:0,timeout:20000",
    "_gpsFilter={ready:false",
    "_pkMetricCache=new Map()",
    "_bestPkMetricMatch(p)",
    "pkm-query",
    "PK métrique 1 m · matchage LGV",
    "pas une localisation au mètre",
    "visibilitychange",
]
for marker in required:
    if marker not in s:
        raise SystemExit(f"ERREUR vérification : marqueur absent : {marker}")

# Contrôle structurel minimal.
if s.count("function startLiveLocation(") != 1:
    raise SystemExit("ERREUR : startLiveLocation dupliquée")
if s.count("function findNearest(") != 1:
    raise SystemExit("ERREUR : findNearest dupliquée")
if s.count("function _finishFindNearest(") != 1:
    raise SystemExit("ERREUR : _finishFindNearest dupliquée")

OUTPUT.write_text(s, encoding="utf-8", newline="\n")
print(f"OK : {OUTPUT} ({VERSION})")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corrige index_v3.html pour privilégier strictement les PK métriques PMTiles."""
from pathlib import Path
import re

P=Path('index_v3.html')
if not P.exists(): raise SystemExit('ERREUR: index_v3.html introuvable')
s=P.read_text(encoding='utf-8')
V='v3.78-pk-metrique'
s=re.sub(r"var APP_VER\s*=\s*'[^']+';",f"var APP_VER = '{V}';",s,count=1)
s=re.sub(r"var V\s*=\s*'[^']+';",f"var V='{V}';",s,count=1)

# Ajoute un index léger des PK métriques rencontrés dans les tuiles chargées.
INSERT=r'''
// ── PK métrique terrain : index des points PMTiles réellement chargés ──
var _pkMetricCache=new Map();
function _pkMetricKey(f){
  var c=f.geometry&&f.geometry.coordinates||[];
  return c.length>=2?c[0].toFixed(7)+'|'+c[1].toFixed(7)+'|'+String(f.properties&&
    (f.properties._pk_label!=null?f.properties._pk_label:f.properties.pk)||''):'';
}
function _refreshPkMetricCache(){
  if(!map.getSource('pk_metric'))return 0;
  try{
    var fs=map.querySourceFeatures('pk_metric',{sourceLayer:'pk_metric'}),added=0;
    fs.forEach(function(f){
      if(!f.geometry||f.geometry.type!=='Point')return;
      var k=_pkMetricKey(f);if(!k)return;
      if(!_pkMetricCache.has(k)){_pkMetricCache.set(k,{props:Object.assign({},f.properties),geom:{type:'Point',coordinates:f.geometry.coordinates.slice()}});added++;}
    });
    return added;
  }catch(e){console.warn('PK metric cache:',e);return 0;}
}
function _nearestPkMetric(lat,lng,maxDistance){
  _refreshPkMetricCache();
  var best=null;
  _pkMetricCache.forEach(function(f){
    var co=f.geom.coordinates,d=_haversine(lat,lng,co[1],co[0]);
    if(d<=maxDistance&&(!best||d<best.d))best={f:f,d:d};
  });
  return best;
}
function _pkMetricLabel(p){
  var raw=p&&p._pk_label!=null?p._pk_label:(p&&p.pk!=null?p.pk:null);
  if(raw==null||raw==='')return '';
  var text=String(raw);
  if(text.indexOf('+')>=0)return text;
  var n=Number(raw);if(!isFinite(n))return text;
  var km=Math.floor(n/1000),m=Math.round(n-km*1000);
  return km+'+'+('000'+m).slice(-3);
}
'''
anchor='// ══════════════════════════════════════════════════════════\n// DETECTION PK / ACCES / PAM LE PLUS PROCHE'
if '_pkMetricCache=new Map()' not in s:
    idx=s.find(anchor)
    if idx<0:raise SystemExit('ERREUR: section detection PK introuvable')
    s=s[:idx]+INSERT+'\n'+s[idx:]

# Remplace findNearest par une phase de chargement forcé des tuiles métriques.
FIND=r'''function findNearest(){
  if(!navigator.geolocation){showNetToast('📍 Géolocalisation non disponible');return;}
  function withPosition(p){
    var lat=p.lat!=null?p.lat:p.coords.latitude,lng=p.lng!=null?p.lng:p.coords.longitude,acc=p.accuracy!=null?p.accuracy:p.coords.accuracy;
    openDetail('📍 Éléments LGV proches','<div style="padding:20px;text-align:center;color:#888"><p style="font-size:14px;margin:0">Chargement des PK métriques…</p><p style="font-size:10px;margin-top:8px;color:#aaa">Zoom technique z20 et lecture PMTiles autour de la position.</p></div>');
    map.jumpTo({center:[lng,lat],zoom:20});
    var tries=0,done=false;
    function finish(){
      if(done)return;tries++;_refreshPkMetricCache();
      var metric=_nearestPkMetric(lat,lng,250);
      if(metric){done=true;_finishFindNearest(lat,lng,acc,metric);return;}
      if(tries<8){setTimeout(finish,500);return;}
      done=true;_finishFindNearest(lat,lng,acc,null);
    }
    map.once('idle',finish);setTimeout(finish,1200);
  }
  if(typeof GPS!=='undefined'&&GPS.last&&Date.now()-(GPS.last.receivedAt||GPS.last.timestamp)<10000){withPosition(GPS.last);return;}
  navigator.geolocation.getCurrentPosition(function(pos){withPosition(pos);},function(e){showNetToast('📍 Position GPS indisponible');},{enableHighAccuracy:true,maximumAge:0,timeout:15000});
}
window.findNearest=findNearest;'''
s,n=re.subn(r'function findNearest\(\)\{.*?\n\}\nwindow\.findNearest=findNearest;',FIND,s,count=1,flags=re.S)
if n!=1:raise SystemExit('ERREUR: findNearest introuvable')

# Remplace l'entête et les niveaux de recherche de _finishFindNearest.
start=s.find('function _finishFindNearest(lat,lng,acc)')
if start<0:
    start=s.find('function _finishFindNearest(lat,lng,acc,metricOverride)')
end=s.find('  /* Détecter branche + côté depuis la voie du PK le plus proche */',start)
if start<0 or end<0:raise SystemExit('ERREUR: _finishFindNearest introuvable')
NEW=r'''function _finishFindNearest(lat,lng,acc,metricOverride){
  var npk=metricOverride||_nearestPkMetric(lat,lng,250),pkPrec=npk?'1 m PMTiles':'';

  /*
   * Règle terrain : ne jamais présenter une interpolation 100 m comme un PK précis.
   * Le PK hectométrique est uniquement un secours explicite si aucune tuile métrique
   * n'est disponible après les tentatives de chargement.
   */
  if(!npk&&_pkHData&&_pkHData.features&&_pkHData.features.length){
    var candidates=[];
    _pkHData.features.forEach(function(f){
      if(!f.geometry||f.geometry.type!=='Point')return;
      var d=_haversine(lat,lng,f.geometry.coordinates[1],f.geometry.coordinates[0]);
      candidates.push({f:{props:f.properties,geom:f.geometry},d:d});
    });
    candidates.sort(function(a,b){return a.d-b.d;});
    if(candidates.length){npk=candidates[0];pkPrec='secours 100 m';}
  }

  if(!npk&&layerData.pk.length){npk=_nearestOf(lat,lng,layerData.pk);if(npk)pkPrec='secours 1 km';}
'''
s=s[:start]+NEW+s[end:]

# Rend le statut de précision très visible dans le panneau.
s=s.replace("html+='<p style=\"font-size:11px;color:#888;margin:0\">📍 GPS : ±'+Math.round(acc)+' m · PK : précision '+pkPrec+voieInfo+'</p>';",
"html+='<p style=\"font-size:11px;color:'+(pkPrec==='1 m PMTiles'?'#2e7d32':'#c62828')+';font-weight:700;margin:0\">📍 GPS : ±'+Math.round(acc)+' m · PK : '+pkPrec+voieInfo+'</p>';" )

# Si le résultat est métrique, normalise le libellé avec la fonction dédiée.
s=s.replace("var rawLbl=p._pk_label!=null?p._pk_label:(p.pk!=null?p.pk:null);",
"var rawLbl=pkPrec==='1 m PMTiles'?_pkMetricLabel(p):(p._pk_label!=null?p._pk_label:(p.pk!=null?p.pk:null));",1)

# Ajoute un avertissement clair pour le fallback.
needle="html+='<div style=\"font-size:18px;font-weight:700;color:#cc0000\">'+lbl+'</div>';"
rep=needle+"\n    if(pkPrec!=='1 m PMTiles')html+='<div style=\"font-size:11px;color:#c62828;font-weight:700;margin-top:3px\">⚠ PK métrique indisponible : valeur de secours, ne pas utiliser comme précision 1 m.</div>';"
s=s.replace(needle,rep,1)

# Contrôle syntaxique basique et marqueurs.
for marker in [V,'_pkMetricCache=new Map()','zoom:20','secours 100 m','PK métrique indisponible']:
    if marker not in s:raise SystemExit('ERREUR verification: '+marker)
P.write_text(s,encoding='utf-8',newline='\n')
print('OK',P,V)

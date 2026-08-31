Put india-archive-bg.png here -- a low-opacity background texture
(faint India silhouette, newspaper/archive grain, sparse muted-rust
dots). Referenced by rag/app.py's PAGE template as
"/static/india-archive-bg.png".

Without it, the page still works -- the CSS background-image rule
just fails silently, leaving the plain navy background with no
texture. Not a crash, just missing decoration until this file exists.

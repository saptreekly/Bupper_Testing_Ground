{pkgs}: {
  deps = [
    pkgs.glibcLocales
    pkgs.geos
    pkgs.spdlog
    pkgs.nlohmann_json
    pkgs.muparserx
    pkgs.fmt
    pkgs.catch2
    pkgs.xsimd
    pkgs.libxcrypt
    pkgs.tk
    pkgs.tcl
    pkgs.qhull
    pkgs.pkg-config
    pkgs.gtk3
    pkgs.gobject-introspection
    pkgs.ghostscript
    pkgs.freetype
    pkgs.ffmpeg-full
    pkgs.cairo
  ];
}

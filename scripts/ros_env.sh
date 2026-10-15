# Общее окружение для скриптов с ROS 2 и окнами (source, не запускать).
# Терминал VS Code из snap подсовывает свои GTK-библиотеки, и rviz2 падает с
# "libpthread.so.0: undefined symbol: __libc_pthread_init". Убираем их.
if [ -n "$SNAP_NAME" ]; then
    unset GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE_FILE GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR \
          GDK_PIXBUF_MODULE_FILE GDK_PIXBUF_MODULEDIR LOCPATH
    [ -n "$XDG_DATA_DIRS_VSCODE_SNAP_ORIG" ] && export XDG_DATA_DIRS="$XDG_DATA_DIRS_VSCODE_SNAP_ORIG"
    [ -n "$XDG_CONFIG_DIRS_VSCODE_SNAP_ORIG" ] && export XDG_CONFIG_DIRS="$XDG_CONFIG_DIRS_VSCODE_SNAP_ORIG"
fi
source /opt/ros/humble/setup.bash

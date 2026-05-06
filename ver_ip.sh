#!/bin/bash
IP=$(hostname -I | awk '{print $1}')
echo -e "\n🌐 La URL de tu servidor en este momento es:\n\n   http://$IP:8000\n\n==============================\n"
read -p "Presiona Enter para cerrar esta ventana..."

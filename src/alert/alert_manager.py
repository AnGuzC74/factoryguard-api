"""
Sistema de Alertas para el pronóstico industrial.
Soporta notificaciones por correo electrónico y Slack.
"""
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import tomllib


class AlertManager:
    def __init__(self, config_path: Path = Path("config.toml")):
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)
        self.alert_config = self.config.get("alertas", {})
        self.email_from = self.alert_config.get("email_from", "")
        self.email_to = self.alert_config.get("email_to", "")
        self.smtp_server = self.alert_config.get("smtp_server", "")
        self.smtp_port = self.alert_config.get("smtp_port", 587)
        self.smtp_user = self.alert_config.get("smtp_user", "")
        self.smtp_password = self.alert_config.get("smtp_password", "")
        self.slack_webhook = self.alert_config.get("slack_webhook_url", "")

    def _get_email_body(self, asset_name: str, status: Dict) -> str:
        return f"""
        🚨 ALERTA DE MANTENIMIENTO PREDICTIVO

        Activo: {asset_name}
        Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        📊 Diagnóstico:
        - RMS Actual: {status.get('rms_actual', 0):.4f} g
        - RMS Máximo Histórico: {status.get('rms_max', 0):.4f} g
        - Frecuencia Dominante: {status.get('frecuencia', 0):.1f} Hz
        - RUL Estimado: {status.get('rul_hours', 0):.1f} horas
        - Zona de Falla: {status.get('zona_falla', 'No definida')}

        🎯 Recomendación:
        {status.get('recomendacion', 'Revisar inmediatamente.')}

        ---
        Sistema de Pronóstico Industrial
        """

    def send_email(self, asset_name: str, status: Dict) -> bool:
        if not all([self.email_from, self.email_to, self.smtp_server]):
            return False
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            msg['Subject'] = f"🚨 ALERTA: {asset_name} - RUL Crítico"
            msg.attach(MIMEText(self._get_email_body(asset_name, status), 'plain'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"[EMAIL] Error: {e}")
            return False

    def send_slack(self, asset_name: str, status: Dict) -> bool:
        if not self.slack_webhook:
            return False
        try:
            mensaje = f"""
🚨 *ALERTA DE MANTENIMIENTO PREDICTIVO*
*Activo:* {asset_name}
*RUL:* {status.get('rul_hours', 0):.1f} horas
*RMS:* {status.get('rms_actual', 0):.4f} g
*Frecuencia:* {status.get('frecuencia', 0):.1f} Hz
*Zona de Falla:* {status.get('zona_falla', 'No definida')}
*Recomendación:* {status.get('recomendacion', 'Revisar inmediatamente.')}
            """
            response = requests.post(
                self.slack_webhook,
                json={"text": mensaje},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"[SLACK] Error: {e}")
            return False

    def send_alert(self, asset_name: str, status: Dict, channels: list = None) -> bool:
        if channels is None:
            channels = ["email", "slack"]
        success = False
        if "email" in channels and self.send_email(asset_name, status):
            success = True
        if "slack" in channels and self.send_slack(asset_name, status):
            success = True
        return success
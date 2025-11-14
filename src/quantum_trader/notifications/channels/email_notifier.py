"""Email notification channel for sending alerts via email.

This module provides functionality to send email notifications using SMTP
with support for HTML formatting, attachments, and template rendering.
"""

import asyncio
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from structlog import get_logger

logger = get_logger(__name__)


class EmailNotifier:
    """Email notification channel.

    Sends email notifications via SMTP with support for HTML content,
    attachments, and retry logic.

    Attributes:
        config: Configuration dictionary
        smtp_host: SMTP server host
        smtp_port: SMTP server port
        smtp_user: SMTP username
        smtp_password: SMTP password
        from_email: Sender email address
        use_tls: Whether to use TLS

    Example:
        >>> config = {
        ...     "smtp_host": "smtp.gmail.com",
        ...     "smtp_port": 587,
        ...     "smtp_user": "notifications@example.com",
        ...     "from_email": "noreply@example.com",
        ...     "default_recipients": ["admin@example.com"]
        ... }
        >>> notifier = EmailNotifier(config)
        >>> await notifier.send_email(
        ...     subject="Alert",
        ...     body="Test alert",
        ...     recipients=["admin@example.com"]
        ... )
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize email notifier.

        Args:
            config: Configuration dictionary containing:
                - smtp_host: SMTP server hostname
                - smtp_port: SMTP server port
                - smtp_user: SMTP username
                - smtp_password: SMTP password (from env)
                - from_email: Sender email address
                - from_name: Sender display name
                - use_tls: Use TLS (default True)
                - use_ssl: Use SSL (default False)
                - default_recipients: List of default recipients
                - retry_attempts: Number of retry attempts
                - retry_delay: Delay between retries in seconds

        Raises:
            ValueError: If required config parameters are missing
        """
        self.config = config
        self._validate_config()

        self.smtp_host = self.config.get("smtp_host", os.getenv("SMTP_HOST"))
        self.smtp_port = int(self.config.get("smtp_port", os.getenv("SMTP_PORT", "587")))
        self.smtp_user = self.config.get("smtp_user", os.getenv("SMTP_USER"))
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")

        self.from_email = self.config.get("from_email", os.getenv("SMTP_FROM_EMAIL"))
        self.from_name = self.config.get(
            "from_name", os.getenv("SMTP_FROM_NAME", "Quantum Trader AI")
        )

        self.use_tls = self.config.get(
            "use_tls", os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        )
        self.use_ssl = self.config.get(
            "use_ssl", os.getenv("SMTP_USE_SSL", "false").lower() == "true"
        )

        self.default_recipients = self.config.get(
            "default_recipients",
            os.getenv("SMTP_DEFAULT_RECIPIENTS", "").split(",")
        )
        self.default_recipients = [r.strip() for r in self.default_recipients if r.strip()]

        self.retry_attempts = int(
            self.config.get("retry_attempts", os.getenv("SMTP_RETRY_ATTEMPTS", "3"))
        )
        self.retry_delay = int(
            self.config.get("retry_delay", os.getenv("SMTP_RETRY_DELAY", "5"))
        )

        # Statistics
        self._emails_sent = 0
        self._emails_failed = 0

        logger.info(
            "EmailNotifier initialized",
            smtp_host=self.smtp_host,
            smtp_port=self.smtp_port,
            from_email=self.from_email,
            default_recipients_count=len(self.default_recipients)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        required_fields = ["smtp_host"]
        missing_fields = []

        for field in required_fields:
            if not self.config.get(field) and not os.getenv(field.upper()):
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(f"Missing required config fields: {missing_fields}")

        logger.debug("Config validation passed")

    def _create_smtp_connection(self) -> smtplib.SMTP:
        """Create SMTP connection.

        Returns:
            SMTP connection object

        Raises:
            Exception: If connection fails
        """
        try:
            if self.use_ssl:
                smtp = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=30)
            else:
                smtp = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30)

            if self.use_tls and not self.use_ssl:
                smtp.starttls()

            if self.smtp_user and self.smtp_password:
                smtp.login(self.smtp_user, self.smtp_password)

            logger.debug("SMTP connection established", host=self.smtp_host)
            return smtp

        except Exception as e:
            logger.error(
                "Failed to create SMTP connection",
                host=self.smtp_host,
                port=self.smtp_port,
                error=str(e),
                exc_info=True
            )
            raise

    async def send_email(
        self,
        subject: str,
        body: str,
        recipients: Optional[List[str]] = None,
        html_body: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None
    ) -> bool:
        """Send an email notification.

        Args:
            subject: Email subject
            body: Plain text email body
            recipients: List of recipient email addresses
            html_body: HTML email body (optional)
            attachments: List of attachment dicts (optional)
            cc: List of CC recipients (optional)
            bcc: List of BCC recipients (optional)

        Returns:
            True if email was sent successfully, False otherwise
        """
        try:
            # Use default recipients if none provided
            if not recipients:
                recipients = self.default_recipients

            if not recipients:
                logger.warning("No recipients specified for email")
                return False

            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = ", ".join(recipients)

            if cc:
                msg["Cc"] = ", ".join(cc)

            # Add plain text body
            msg.attach(MIMEText(body, "plain"))

            # Add HTML body if provided
            if html_body:
                msg.attach(MIMEText(html_body, "html"))

            # Add attachments if provided
            if attachments:
                for attachment in attachments:
                    await self._add_attachment(msg, attachment)

            # Prepare recipient list
            all_recipients = recipients.copy()
            if cc:
                all_recipients.extend(cc)
            if bcc:
                all_recipients.extend(bcc)

            # Send email with retry logic
            for attempt in range(self.retry_attempts):
                try:
                    # Run SMTP operations in thread pool
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        self._send_via_smtp,
                        msg,
                        all_recipients
                    )

                    self._emails_sent += 1

                    logger.info(
                        "Email sent successfully",
                        subject=subject,
                        recipients_count=len(recipients),
                        attempt=attempt + 1
                    )

                    return True

                except Exception as e:
                    if attempt < self.retry_attempts - 1:
                        logger.warning(
                            "Email send attempt failed, retrying",
                            attempt=attempt + 1,
                            error=str(e)
                        )
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        raise

        except Exception as e:
            self._emails_failed += 1
            logger.error(
                "Failed to send email",
                subject=subject,
                recipients_count=len(recipients) if recipients else 0,
                error=str(e),
                exc_info=True
            )
            return False

    def _send_via_smtp(self, msg: MIMEMultipart, recipients: List[str]) -> None:
        """Send email via SMTP (blocking operation).

        Args:
            msg: Email message
            recipients: List of recipients
        """
        smtp = self._create_smtp_connection()
        try:
            smtp.send_message(msg, to_addrs=recipients)
        finally:
            smtp.quit()

    async def _add_attachment(
        self,
        msg: MIMEMultipart,
        attachment: Dict[str, Any]
    ) -> None:
        """Add attachment to email message.

        Args:
            msg: Email message
            attachment: Attachment dictionary with:
                - filename: Attachment filename
                - content: Attachment content (bytes)
                - content_type: MIME content type (optional)
        """
        try:
            filename = attachment.get("filename", "attachment")
            content = attachment.get("content", b"")

            part = MIMEBase("application", "octet-stream")
            part.set_payload(content)
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {filename}"
            )

            msg.attach(part)

            logger.debug("Attachment added", filename=filename)

        except Exception as e:
            logger.error(
                "Failed to add attachment",
                filename=attachment.get("filename"),
                error=str(e),
                exc_info=True
            )

    async def send_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send an alert notification via email.

        Args:
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
            data: Additional alert data

        Returns:
            True if alert was sent successfully, False otherwise
        """
        try:
            # Format subject
            subject = f"[{severity.upper()}] {alert_type}"

            # Format body
            body = f"{message}\n\n"
            body += f"Alert Type: {alert_type}\n"
            body += f"Severity: {severity}\n"
            body += f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n"

            if data:
                body += "\nAdditional Information:\n"
                for key, value in data.items():
                    body += f"  {key}: {value}\n"

            # Create HTML version
            html_body = self._create_html_alert(alert_type, severity, message, data)

            # Send email
            return await self.send_email(
                subject=subject,
                body=body,
                html_body=html_body
            )

        except Exception as e:
            logger.error(
                "Failed to send alert email",
                alert_type=alert_type,
                severity=severity,
                error=str(e),
                exc_info=True
            )
            return False

    def _create_html_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create HTML formatted alert email.

        Args:
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
            data: Additional alert data

        Returns:
            HTML string
        """
        # Severity color mapping
        severity_colors = {
            "CRITICAL": "#dc3545",
            "HIGH": "#fd7e14",
            "MEDIUM": "#ffc107",
            "LOW": "#28a745",
            "INFO": "#17a2b8",
        }

        color = severity_colors.get(severity.upper(), "#6c757d")

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .alert-box {{
                    border-left: 4px solid {color};
                    padding: 15px;
                    background-color: #f8f9fa;
                    margin: 20px 0;
                }}
                .alert-header {{
                    color: {color};
                    font-size: 18px;
                    font-weight: bold;
                }}
                .alert-message {{ margin: 10px 0; }}
                .alert-details {{
                    background-color: white;
                    padding: 10px;
                    margin-top: 10px;
                    border-radius: 4px;
                }}
                .detail-row {{ margin: 5px 0; }}
                .detail-label {{ font-weight: bold; color: #495057; }}
            </style>
        </head>
        <body>
            <div class="alert-box">
                <div class="alert-header">[{severity.upper()}] {alert_type}</div>
                <div class="alert-message">{message}</div>
                <div class="alert-details">
                    <div class="detail-row">
                        <span class="detail-label">Timestamp:</span>
                        {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
                    </div>
        """

        if data:
            html += '<div class="detail-row"><span class="detail-label">Additional Information:</span></div>'
            for key, value in data.items():
                html += f'<div class="detail-row" style="margin-left: 20px;">{key}: {value}</div>'

        html += """
                </div>
            </div>
        </body>
        </html>
        """

        return html

    async def test_connection(self) -> bool:
        """Test SMTP connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._test_smtp_connection
            )

            logger.info("SMTP connection test successful")
            return True

        except Exception as e:
            logger.error(
                "SMTP connection test failed",
                error=str(e),
                exc_info=True
            )
            return False

    def _test_smtp_connection(self) -> None:
        """Test SMTP connection (blocking operation)."""
        smtp = self._create_smtp_connection()
        smtp.quit()

    def get_stats(self) -> Dict[str, Any]:
        """Get email notifier statistics.

        Returns:
            Dictionary with notifier stats
        """
        success_rate = 0.0
        total = self._emails_sent + self._emails_failed
        if total > 0:
            success_rate = (self._emails_sent / total) * 100

        return {
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "from_email": self.from_email,
            "default_recipients_count": len(self.default_recipients),
            "emails_sent": self._emails_sent,
            "emails_failed": self._emails_failed,
            "success_rate_percent": f"{success_rate:.2f}",
        }

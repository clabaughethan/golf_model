"""
Send picks/summary email via Gmail SMTP.

Setup (one-time):
  1. Go to myaccount.google.com -> Security -> 2-Step Verification -> App passwords
  2. Generate an app password for "Mail"
  3. Set environment variables:
       GMAIL_USER=you@gmail.com
       GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

  NOTIFY_TO defaults to GMAIL_USER if not set.
"""

import os
import smtplib
from email.mime.text import MIMEText


def send_email(subject: str, body: str) -> bool:
    gmail_user = os.environ.get("GMAIL_USER", "")
    app_password = os.environ.get("GMAIL_APP_PASSWORD", "")
    to_addr = os.environ.get("NOTIFY_TO", gmail_user)

    if not gmail_user or not app_password:
        print("  Email skipped — GMAIL_USER / GMAIL_APP_PASSWORD not set.")
        return False

    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = to_addr

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, app_password)
            server.sendmail(gmail_user, to_addr, msg.as_string())
        print(f"  Email sent to {to_addr}")
        return True
    except Exception as e:
        print(f"  Email failed: {e}")
        return False


def send_picks(picks_text: str, tournament: str, date: str):
    subject = f"Golf Model Picks — {tournament} ({date})"
    print(f"\n  Sending picks email...")
    send_email(subject, picks_text)

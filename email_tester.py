import time
from datetime import datetime
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
import yaml
import random


settings = yaml.safe_load(open("config.yaml"))
from_addr = settings['from_addr']
to_addr = settings['to_addr']
password = settings['password']

# Emailer section
# ---------------------------------------------------------
subject = 'Found This'
body = ('The A.I. script has found a nothing.')
msg = EmailMessage()
msg.add_header('from', from_addr)
msg.add_header('to', ', '.join(to_addr))
msg.add_header('subject', subject)

# Saves images
now_sub = datetime.now()
now_sub = now_sub.strftime("%Y_%m_%d-%H_%M_%S")

# For naming images
img_1_name = now_sub + "-image_1"
img_2_name = now_sub + "-image_2"
img_3_name = now_sub + "-image_3"

attachment_cid_1 = make_msgid()
attachment_cid_2 = make_msgid()
attachment_cid_3 = make_msgid()

msg.set_content(
    '<b>%s</b><br/><br/><br/>' % (
        body),
    'html'
)

try:
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(from_addr, password)
    server.send_message(msg, from_addr=from_addr, to_addrs=to_addr)
    server.quit()
except:
    print('Something went wrong...')
# =============================================================================

print("\nDone!")
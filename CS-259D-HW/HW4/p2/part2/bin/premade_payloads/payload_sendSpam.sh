#! /bin/sh

emailUrl="https://dl.dropboxusercontent.com/u/62123265/CS%20259d/spam_emails/email_downloadMoreRam.txt"

emailRecipiets=( "gavilan@stanford.edu" "gsgalloway@hotmail.com" "galloway.gavilan@stanford.edu" )

emailPath="/tmp/email.txt"

# Download the email to send
wget $emailUrl --no-check-certificate -O $emailPath > /dev/null 2>&1
chmod 777 $emailPath > /dev/null 2>&1

# Send the emails
for recipient in "${emailRecipiets[@]}"
do
	echo "Emailing $recipient"
	sendmail $recipient < $emailPath
done

rm $emailPath > /dev/null 2>&1
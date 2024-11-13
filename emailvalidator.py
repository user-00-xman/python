PERMITS_RECIPIENT = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._'
PERMITS_DOMAIN = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
VALID_TLDS = {'com', 'net', 'org', 'tech'}

def validate(email):
    if email.count('@') != 1:
        return False

    recipient_name, domain_and_tld = email.split('@')
    
    # Validate recipient name
    if len(recipient_name) > 24 or len(recipient_name) < 3:
        return False
    for c in recipient_name:
        if c not in PERMITS_RECIPIENT:
            return False
    if recipient_name[0] in '._' or recipient_name[-1] in '._':
        return False

    # Validate domain and TLD
    domain_parts = domain_and_tld.split('.')
    if len(domain_parts) != 2:
        return False
    
    domain_name, tld = domain_parts

    # Validate domain name
    if len(domain_name) > 12 or len(domain_name) < 3:
        return False
    for c in domain_name:
        if c not in PERMITS_DOMAIN:
            return False
    if domain_name[0] == '-' or domain_name[-1] == '-':
        return False

    # Validate TLD
    if tld not in VALID_TLDS:
        return False

    return True

if __name__ == "__main__":
    print("Enter email:")
    email = input()
    validate_email = validate(email)
    if validate_email == 1:
        print("Email is valid")
    else:
        print("Email is invalid")



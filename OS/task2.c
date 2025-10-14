#include <stdio.h>
#include <stdlib.h>

void showMenu() {
    printf("\n--- User and Group Manager ---\n");
    printf("1. Add User\n");
    printf("2. Delete User\n");
    printf("3. Add Group\n");
    printf("4. Delete Group\n");
    printf("5. Modify User Info\n");
    printf("6. Change Account Info\n");
    printf("7. Assign User to Group\n");
    printf("8. Exit\n");
    printf("Choose an option: ");
}

int main() {
    int choice, subChoice, days;
    char username[50], groupname[50], newval[100], command[256];

    while (1) {
        showMenu();
        scanf("%d", &choice);
        getchar();

        switch (choice) {
            case 1:
                printf("Enter username to add: ");
                scanf("%s", username);
                
                printf("Enter password: ");
                system("stty -echo");
                scanf("%s", newval);
                system("stty echo");
                printf("\n");
                
                snprintf(command, sizeof(command), "sudo useradd %s", username);
                system(command);
                snprintf(command, sizeof(command), "echo \"%s:%s\" | sudo chpasswd", username, newval);
                system(command);
                printf("User %s added successfully\n", username);
                break;

            case 2:
                printf("Enter username to delete: ");
                scanf("%s", username);
                snprintf(command, sizeof(command), "sudo userdel %s", username);
                system(command);
                printf("User %s deleted\n", username);
                break;

            case 3:
                printf("Enter group name to add: ");
                scanf("%s", groupname);
                snprintf(command, sizeof(command), "sudo groupadd %s", groupname);
                system(command);
                printf("Group %s added\n", groupname);
                break;

            case 4:
                printf("Enter group name to delete: ");
                scanf("%s", groupname);
                snprintf(command, sizeof(command), "sudo groupdel %s", groupname);
                system(command);
                printf("Group %s deleted\n", groupname);
                break;

            case 5:
                printf("Enter username to modify: ");
                scanf("%s", username);
                printf("1. Change password\n2. Change shell\n3. Change home directory\n4. Change username\nChoose: ");
                scanf("%d", &subChoice);
                getchar();
                switch (subChoice) {
                    case 1:
                        system("stty -echo");
                        printf("Enter new password: ");
                        scanf("%s", newval);
                        system("stty echo");
                        printf("\n");
                        
                        snprintf(command, sizeof(command), "echo \"%s:%s\" | sudo chpasswd", username, newval);
                        system(command);
                        printf("Password changed for %s\n", username);
                        break;
                        
                    case 2:
                        printf("Enter new shell: ");
                        scanf("%s", newval);
                        snprintf(command, sizeof(command), "sudo usermod -s %s %s", newval, username);
                        system(command);
                        printf("Shell changed for %s\n", username);
                        break;
                        
                    case 3:
                        printf("Enter new home directory: ");
                        scanf("%s", newval);
                        snprintf(command, sizeof(command), "sudo usermod -m -d %s %s", newval, username);
                        system(command);
                        printf("Home directory changed for %s\n", username);
                        break;
                        
                    case 4:
                        printf("Enter new username: ");
                        scanf("%s", newval);
                        
                        snprintf(command, sizeof(command), "sudo usermod -l %s %s", newval, username);
                        system(command);
                        
                        snprintf(command, sizeof(command), "sudo usermod -m -d /home/%s %s", newval, newval);
                        system(command);
                        
                        snprintf(command, sizeof(command), "sudo groupmod -n %s %s 2>/dev/null || true", newval, username);
                        system(command);
                        
                        printf("Username changed from %s to %s\n", username, newval);
                        break;
                        
                    default:
                        printf("Invalid option.\n");
                        continue;
                }
                break;

            case 6:
                printf("Enter username to modify account info: ");
                scanf("%s", username);
                printf("1. Set password expiry (in days)\n2. Set minimum password age\n3. Set maximum password age\nChoose: ");
                scanf("%d", &subChoice);
                getchar();
                switch (subChoice) {
                    case 1:
                        printf("Enter expiry in days: ");
                        scanf("%d", &days);
                        snprintf(command, sizeof(command), "sudo chage -E $(date -d \"+%d days\" +%%Y-%%m-%%d) %s", days, username);
                        system(command);
                        printf("Password expiry set to %d days for %s\n", days, username);
                        break;
                    case 2:
                        printf("Enter minimum password age: ");
                        scanf("%d", &days);
                        snprintf(command, sizeof(command), "sudo chage -m %d %s", days, username);
                        system(command);
                        printf("Minimum password age set to %d days for %s\n", days, username);
                        break;
                    case 3:
                        printf("Enter maximum password age: ");
                        scanf("%d", &days);
                        snprintf(command, sizeof(command), "sudo chage -M %d %s", days, username);
                        system(command);
                        printf("Maximum password age set to %d days for %s\n", days, username);
                        break;
                    default:
                        printf("Invalid option.\n");
                        continue;
                }
                break;

            case 7:
                printf("Enter username: ");
                scanf("%s", username);
                printf("Enter group to assign: ");
                scanf("%s", groupname);
                snprintf(command, sizeof(command), "sudo usermod -a -G %s %s", groupname, username);
                system(command);
                printf("User %s added to group %s\n", username, groupname);
                break;

            case 8:
                printf("Exiting...\n");
                return 0;

            default:
                printf("Invalid choice.\n");
        }
    }
    return 0;
}
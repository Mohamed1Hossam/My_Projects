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
        getchar(); // Clear newline

        switch (choice) {
            case 1:
                printf("Enter username to add: ");
                scanf("%s", username);
                printf("Enter password: ");
                scanf("%s", newval);
                snprintf(command, sizeof(command), "sudo useradd %s", username);
                system(command);
                snprintf(command, sizeof(command), "echo \"%s:%s\" | sudo chpasswd", username, newval);
                system(command);
                break;

            case 2:
                printf("Enter username to delete: ");
                scanf("%s", username);
                snprintf(command, sizeof(command), "sudo userdel %s", username);
                system(command);
                break;

            case 3:
                printf("Enter group name to add: ");
                scanf("%s", groupname);
                snprintf(command, sizeof(command), "sudo groupadd %s", groupname);
                system(command);
                break;

            case 4:
                printf("Enter group name to delete: ");
                scanf("%s", groupname);
                snprintf(command, sizeof(command), "sudo groupdel %s", groupname);
                system(command);
                break;

            case 5:
                printf("Enter username to modify: ");
                scanf("%s", username);
                printf("1. Change password\n2. Change shell\n3. Change home directory\n4. Change username\nChoose: ");
                scanf("%d", &subChoice);
                getchar();
                switch (subChoice) {
                    case 1:
                        snprintf(command, sizeof(command), "sudo passwd %s", username);
                        break;
                    case 2:
                        printf("Enter new shell: ");
                        scanf("%s", newval);
                        snprintf(command, sizeof(command), "sudo usermod -s %s %s", newval, username);
                        break;
                    case 3:
                        printf("Enter new home directory: ");
                        scanf("%s", newval);
                        snprintf(command, sizeof(command), "sudo usermod -d %s -m %s", newval, username);
                        break;
                    case 4:
                        printf("Enter new username: ");
                        scanf("%s", newval);
                        snprintf(command, sizeof(command), "sudo usermod -l %s %s", newval, username);
                        break;
                    default:
                        printf("Invalid option.\n");
                        continue;
                }
                system(command);
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
                        snprintf(command, sizeof(command), "sudo chage -M %d %s", days, username);
                        break;
                    case 2:
                        printf("Enter minimum password age: ");
                        scanf("%d", &days);
                        snprintf(command, sizeof(command), "sudo chage -m %d %s", days, username);
                        break;
                    case 3:
                        printf("Enter maximum password age: ");
                        scanf("%d", &days);
                        snprintf(command, sizeof(command), "sudo chage -M %d %s", days, username);
                        break;
                    default:
                        printf("Invalid option.\n");
                        continue;
                }
                system(command);
                break;

            case 7:
                printf("Enter username: ");
                scanf("%s", username);
                printf("Enter group to assign: ");
                scanf("%s", groupname);
                snprintf(command, sizeof(command), "sudo usermod -a -G %s %s", groupname, username);
                system(command);
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
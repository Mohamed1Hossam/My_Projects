#include <gtk/gtk.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>

static GtkWidget *entryUsername, *entryPassword, *entryNewUsername, *entryNewPassword, *entryGroupname, *entryDays;
static GtkWidget *resultLabel;

// Helper function to execute shell commands safely and check the return status
int execute_command(const char *cmd) {
    int status = system(cmd);
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return -1;  // Command failed to execute
}

// Helper function to check if a group exists
int check_group_exists(const char *groupname) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "getent group %s > /dev/null 2>&1", groupname);
    return execute_command(cmd) == 0;
}

// Helper function to check if a user exists
int check_user_exists(const char *username) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "id %s > /dev/null 2>&1", username);
    return execute_command(cmd) == 0;
}

void addUser(GtkWidget *widget, gpointer data) {
    const char *username = gtk_entry_get_text(GTK_ENTRY(entryUsername));
    const char *password = gtk_entry_get_text(GTK_ENTRY(entryPassword));
    char cmd[256];
    int ret;

    // Check if inputs are empty
    if (strlen(username) == 0 || strlen(password) == 0) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Username and password cannot be empty!");
        return;
    }

    // Check if the user already exists
    if (check_user_exists(username)) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: User already exists!");
        return;
    }

    // Add user without specifying a group (will use default)
    snprintf(cmd, sizeof(cmd), "sudo useradd -m %s", username);
    ret = execute_command(cmd);
    if (ret != 0) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Failed to add user!");
        return;
    }

    // Set password for the user using chpasswd
    snprintf(cmd, sizeof(cmd), "echo \"%s:%s\" | sudo chpasswd", username, password);
    ret = execute_command(cmd);
    if (ret != 0) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Failed to set password!");
        return;
    }

    gtk_label_set_text(GTK_LABEL(resultLabel), "User added successfully!");
}

void changeUserInfo(GtkWidget *widget, gpointer data) {
    const char *username = gtk_entry_get_text(GTK_ENTRY(entryUsername));
    const char *newUsername = gtk_entry_get_text(GTK_ENTRY(entryNewUsername));
    const char *newPassword = gtk_entry_get_text(GTK_ENTRY(entryNewPassword));
    char cmd[256];
    int ret;

    // Check if username is empty
    if (strlen(username) == 0) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Current username cannot be empty!");
        return;
    }

    // Check if the user exists
    if (!check_user_exists(username)) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: User does not exist!");
        return;
    }

    // Change username if new username is provided and different
    if (strlen(newUsername) > 0 && strcmp(username, newUsername) != 0) {
        // Check if the new username already exists
        if (check_user_exists(newUsername)) {
            gtk_label_set_text(GTK_LABEL(resultLabel), "Error: New username already exists!");
            return;
        }

        snprintf(cmd, sizeof(cmd), "sudo usermod -l %s %s", newUsername, username);
        ret = execute_command(cmd);
        if (ret != 0) {
            gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Failed to change username!");
            return;
        }
    }

    // Change password if new password is provided
    if (strlen(newPassword) > 0) {
        // Use the new username if it was changed, otherwise use the original
        const char *userToChangePass = (strlen(newUsername) > 0) ? newUsername : username;
        snprintf(cmd, sizeof(cmd), "echo \"%s:%s\" | sudo chpasswd", userToChangePass, newPassword);
        ret = execute_command(cmd);
        if (ret != 0) {
            gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Failed to change password!");
            return;
        }
    }

    gtk_label_set_text(GTK_LABEL(resultLabel), "User information updated successfully!");
}

void deleteUser(GtkWidget *widget, gpointer data) {
    const char *username = gtk_entry_get_text(GTK_ENTRY(entryUsername));
    char cmd[256];
    int ret;
    
    // Check if username is empty
    if (strlen(username) == 0) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Username cannot be empty!");
        return;
    }
    
    // Check if the user exists
    if (!check_user_exists(username)) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: User does not exist!");
        return;
    }
    
    snprintf(cmd, sizeof(cmd), "sudo userdel -r %s", username);  // Also remove user's home directory
    ret = execute_command(cmd);
    if (ret != 0) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Failed to delete user!");
        return;
    }
    
    gtk_label_set_text(GTK_LABEL(resultLabel), "User deleted successfully!");
}

void addGroup(GtkWidget *widget, gpointer data) {
    const char *groupname = gtk_entry_get_text(GTK_ENTRY(entryGroupname));
    char cmd[256];
    int ret;

    // Check if groupname is empty
    if (strlen(groupname) == 0) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Groupname cannot be empty!");
        return;
    }
    
    // Check if the group already exists
    if (check_group_exists(groupname)) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Group already exists!");
        return;
    }

    snprintf(cmd, sizeof(cmd), "sudo groupadd %s", groupname);
    ret = execute_command(cmd);
    if (ret != 0) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Failed to add group!");
        return;
    }

    gtk_label_set_text(GTK_LABEL(resultLabel), "Group added successfully!");
}

void deleteGroup(GtkWidget *widget, gpointer data) {
    const char *groupname = gtk_entry_get_text(GTK_ENTRY(entryGroupname));
    char cmd[256];
    int ret;
    
    // Check if groupname is empty
    if (strlen(groupname) == 0) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Groupname cannot be empty!");
        return;
    }
    
    // Check if the group exists
    if (!check_group_exists(groupname)) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Group does not exist!");
        return;
    }
    
    snprintf(cmd, sizeof(cmd), "sudo groupdel %s", groupname);
    ret = execute_command(cmd);
    if (ret != 0) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Failed to delete group! It may be a primary group for some users.");
        return;
    }
    
    gtk_label_set_text(GTK_LABEL(resultLabel), "Group deleted successfully!");
}

void assignUserToGroup(GtkWidget *widget, gpointer data) {
    const char *username = gtk_entry_get_text(GTK_ENTRY(entryUsername));
    const char *groupname = gtk_entry_get_text(GTK_ENTRY(entryGroupname));
    char cmd[256];
    int ret;
    
    // Check if inputs are empty
    if (strlen(username) == 0 || strlen(groupname) == 0) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Username and groupname cannot be empty!");
        return;
    }
    
    // Check if the user exists
    if (!check_user_exists(username)) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: User does not exist!");
        return;
    }
    
    // Check if the group exists
    if (!check_group_exists(groupname)) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Group does not exist!");
        return;
    }
    
    snprintf(cmd, sizeof(cmd), "sudo usermod -aG %s %s", groupname, username);
    ret = execute_command(cmd);
    if (ret != 0) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Failed to assign user to group!");
        return;
    }

    gtk_label_set_text(GTK_LABEL(resultLabel), "User assigned to group successfully!");
}

void changeAccountInfo(GtkWidget *widget, gpointer data) {
    const char *username = gtk_entry_get_text(GTK_ENTRY(entryUsername));
    const char *days_str = gtk_entry_get_text(GTK_ENTRY(entryDays));
    char cmd[256];
    int ret;
    
    // Check if inputs are empty
    if (strlen(username) == 0 || strlen(days_str) == 0) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Username and days until expiration cannot be empty!");
        return;
    }
    
    // Check if the user exists
    if (!check_user_exists(username)) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: User does not exist!");
        return;
    }
    
    // Check if days_str is a valid number
    for (int i = 0; days_str[i] != '\0'; i++) {
        if (days_str[i] < '0' || days_str[i] > '9') {
            gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Days until expiration must be a number!");
            return;
        }
    }
    
    snprintf(cmd, sizeof(cmd), "sudo chage -M %s %s", days_str, username);
    ret = execute_command(cmd);
    if (ret != 0) {
        gtk_label_set_text(GTK_LABEL(resultLabel), "Error: Failed to change account info!");
        return;
    }

    gtk_label_set_text(GTK_LABEL(resultLabel), "Account info (password expiration) updated successfully!");
}

void showMainWindow(GtkWidget *widget, gpointer data) {
    GtkWidget *window, *grid, *label, *addUserButton, *changeUserButton, *deleteUserButton;
    GtkWidget *addGroupButton, *deleteGroupButton, *assignUserButton, *changeAccountButton;

    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "User and Group Manager");
    gtk_window_set_default_size(GTK_WINDOW(window), 400, 500);
    gtk_container_set_border_width(GTK_CONTAINER(window), 10);

    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

    grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(grid), 5);
    gtk_grid_set_column_spacing(GTK_GRID(grid), 5);
    gtk_container_add(GTK_CONTAINER(window), grid);

    // Username Label and Entry
    label = gtk_label_new("Username:");
    gtk_grid_attach(GTK_GRID(grid), label, 0, 0, 1, 1);
    entryUsername = gtk_entry_new();
    gtk_grid_attach(GTK_GRID(grid), entryUsername, 1, 0, 1, 1);

    // Password Label and Entry (for addUser)
    label = gtk_label_new("Password:");
    gtk_grid_attach(GTK_GRID(grid), label, 0, 1, 1, 1);
    entryPassword = gtk_entry_new();
    gtk_entry_set_visibility(GTK_ENTRY(entryPassword), FALSE);  // Hide password text
    gtk_grid_attach(GTK_GRID(grid), entryPassword, 1, 1, 1, 1);

    // New Username Label and Entry (for changeUserInfo)
    label = gtk_label_new("New Username:");
    gtk_grid_attach(GTK_GRID(grid), label, 0, 2, 1, 1);
    entryNewUsername = gtk_entry_new();
    gtk_grid_attach(GTK_GRID(grid), entryNewUsername, 1, 2, 1, 1);

    // New Password Label and Entry (for changeUserInfo)
    label = gtk_label_new("New Password:");
    gtk_grid_attach(GTK_GRID(grid), label, 0, 3, 1, 1);
    entryNewPassword = gtk_entry_new();
    gtk_entry_set_visibility(GTK_ENTRY(entryNewPassword), FALSE);  // Hide password text
    gtk_grid_attach(GTK_GRID(grid), entryNewPassword, 1, 3, 1, 1);

    // Groupname Label and Entry (for addGroup, deleteGroup, assignUserToGroup)
    label = gtk_label_new("Groupname:");
    gtk_grid_attach(GTK_GRID(grid), label, 0, 4, 1, 1);
    entryGroupname = gtk_entry_new();
    gtk_grid_attach(GTK_GRID(grid), entryGroupname, 1, 4, 1, 1);

    // Password Expiration Label and Entry (for changeAccountInfo)
    label = gtk_label_new("Days Until Expiration:");
    gtk_grid_attach(GTK_GRID(grid), label, 0, 5, 1, 1);
    entryDays = gtk_entry_new();
    gtk_grid_attach(GTK_GRID(grid), entryDays, 1, 5, 1, 1);

    // Result Label
    resultLabel = gtk_label_new("");
    gtk_grid_attach(GTK_GRID(grid), resultLabel, 0, 6, 2, 1);

    // Add User Button
    addUserButton = gtk_button_new_with_label("Add User");
    g_signal_connect(addUserButton, "clicked", G_CALLBACK(addUser), NULL);
    gtk_grid_attach(GTK_GRID(grid), addUserButton, 0, 7, 2, 1);

    // Change User Information Button
    changeUserButton = gtk_button_new_with_label("Change User Information");
    g_signal_connect(changeUserButton, "clicked", G_CALLBACK(changeUserInfo), NULL);
    gtk_grid_attach(GTK_GRID(grid), changeUserButton, 0, 8, 2, 1);

    // Delete User Button
    deleteUserButton = gtk_button_new_with_label("Delete User");
    g_signal_connect(deleteUserButton, "clicked", G_CALLBACK(deleteUser), NULL);
    gtk_grid_attach(GTK_GRID(grid), deleteUserButton, 0, 9, 2, 1);

    // Add Group Button
    addGroupButton = gtk_button_new_with_label("Add Group");
    g_signal_connect(addGroupButton, "clicked", G_CALLBACK(addGroup), NULL);
    gtk_grid_attach(GTK_GRID(grid), addGroupButton, 0, 10, 2, 1);

    // Delete Group Button
    deleteGroupButton = gtk_button_new_with_label("Delete Group");
    g_signal_connect(deleteGroupButton, "clicked", G_CALLBACK(deleteGroup), NULL);
    gtk_grid_attach(GTK_GRID(grid), deleteGroupButton, 0, 11, 2, 1);

    // Assign User to Group Button
    assignUserButton = gtk_button_new_with_label("Assign User to Group");
    g_signal_connect(assignUserButton, "clicked", G_CALLBACK(assignUserToGroup), NULL);
    gtk_grid_attach(GTK_GRID(grid), assignUserButton, 0, 12, 2, 1);

    // Change Account Info Button
    changeAccountButton = gtk_button_new_with_label("Change Account Info");
    g_signal_connect(changeAccountButton, "clicked", G_CALLBACK(changeAccountInfo), NULL);
    gtk_grid_attach(GTK_GRID(grid), changeAccountButton, 0, 13, 2, 1);

    gtk_widget_show_all(window);
}

int main(int argc, char *argv[]) {
    gtk_init(&argc, &argv);

    showMainWindow(NULL, NULL);

    gtk_main();
    return 0;
}
